import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers # type: ignore
import tensorflow_models as tfm # type: ignore

from tfprocess import TFProcess

CONFIG_FILE_PATH = 'configs/768x15x24h.yaml'

def get_sinusoidal_positional_encoding(max_positions=500, d_model=768):
    position = np.arange(max_positions)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pos_encoding = np.zeros((max_positions, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)
    
    return tf.constant(pos_encoding, dtype=tf.float32)

class GameAggregateViT(tf.keras.Model):
    def __init__(
        self,
        move_feature_dim,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        max_moves,
        **kwargs
    ):
        super(GameAggregateViT, self).__init__(**kwargs)
        
        self.move_feature_dim = move_feature_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.max_moves = max_moves
        self.paired_max_moves = max_moves // 2
        if max_moves % 2 != 0:
            raise ValueError(f"max_moves must be even for before/after pairing, got {max_moves}")

        with open(CONFIG_FILE_PATH, 'r') as file:
            cfg = yaml.safe_load(file)
        
        tfp = TFProcess(cfg)
        tfp.init_net(use_heads=False)
        tfp.restore()
        
        # ONLY extract body, not heads. Freeze LC0 weights so stylometry
        # training only updates pair_projection / ViT / elo heads (avoids
        # GradientTape OOM from 15-layer encoder activations).
        input_var = tf.keras.Input(shape=(112, 8, 8))
        assert isinstance(cfg, dict)
        self.move_compress = None
        if cfg['model'].get('encoder_layers', 0) > 0:
            embedding_size = cfg['model'].get('embedding_size', cfg['model'].get('filters'))
            flow, _ = tfp.create_encoder_body(input_var, embedding_size)
            flow = tf.keras.layers.GlobalAveragePooling1D()(flow)
            self.move_projection = tf.keras.Model(inputs=input_var, outputs=flow)
            if flow.shape[-1] != self.hidden_dim:
              self.move_compress = layers.Dense(units=self.hidden_dim, activation='relu')
        else:
            filters = cfg['model']['filters']
            self.move_projection = tfp.model
            if filters * 8 * 8 != self.hidden_dim:
              print("Stylo ViT Using compression layer")
              self.move_compress = tf.keras.Sequential([
                  layers.Flatten(),
                  layers.Dense(units=self.hidden_dim, activation='relu'),
              ])

        self.move_projection.trainable = False
        print("Frozen LC0 move_projection body (stylometry trains ViT + heads only)")
        # Set by GameOutcomePredictor.train_step to tape.stop_recording so body
        # forwards are not recorded on GradientTape (avoids activation OOM).
        self._stop_recording_ctx = None

        self.pair_projection = layers.Dense(units=hidden_dim, activation='relu')

        # BiasAdd GPU kernels multiply N*H*W*C as int32 (limit ~43k positions at
        # emb=768). The 15-layer encoder is the tighter bound: LC0 trains this net
        # at microbatch 512 (batch 2048 / 4 splits). Larger chunks OOM and cuDNN
        # fails with "No DNN support for stream" on fused LayerNorm/FusedBatchNorm.
        self.move_proj_chunk_size = 512

        #sin position encoding like paper recommended
        self.positional_encoding = get_sinusoidal_positional_encoding(
            max_positions=self.paired_max_moves,
            d_model=hidden_dim
        )
        
        # Create input specs for VisionTransformer
        # VisionTransformer expects (batch, height, width, channels) input in 4D
        # Use paired_max_moves as the height dimension.
        input_specs = tf.keras.layers.InputSpec(shape=[None, self.paired_max_moves, 1, hidden_dim])
        
        self.vit = tfm.vision.backbones.VisionTransformer(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_dim,
            mlp_dim=mlp_dim,
            input_specs=input_specs,
            patch_size=1,  # Since we already have embeddings, no patching needed
        )
    
    def _project_moves(self, moves_21, training):
        """Pad to 112 planes and run move_projection in chunks.

        Two limits force chunking:
        1) BiasAdd GPU kernels multiply N*H*W*C as int32 (overflow at ~43k
           positions for emb=768, e.g. 32*20*100 → 64000).
        2) Encoder activations (15 layers, 24-head smolgen) OOM well before that;
           keep chunks near LC0's microbatch (512).
        """
        num_positions = moves_21.shape[0]
        chunk_size = self.move_proj_chunk_size
        d_type = moves_21.dtype

        def _project_chunk(chunk_21):
            n = tf.shape(chunk_21)[0]
            padded = tf.concat([
                chunk_21[:, :13],
                tf.zeros((n, 91, 8, 8), dtype=d_type),
                chunk_21[:, -8:],
            ], axis=1)

            def _run_body():
              # Inference-only on frozen LC0 body (no BN updates).
              return self.move_projection(padded, training=False)

            # When a GradientTape is active, pause recording so encoder
            # activations are not retained for backprop (trainable=False and
            # stop_gradient alone still OOM — body ops are recorded first).
            if self._stop_recording_ctx is not None:
              with self._stop_recording_ctx():
                out = _run_body()
            else:
              out = _run_body()
            out = tf.stop_gradient(out)
            if self.move_compress is not None:
              out = self.move_compress(out, training=training)
            return tf.reshape(out, [-1, self.hidden_dim])

        # Eager path (train_stylometry uses run_eagerly=True).
        if num_positions is not None:
            n = int(num_positions)
            if n == 0:
                return tf.zeros((0, self.hidden_dim), dtype=tf.float32)
            chunks = [
                _project_chunk(moves_21[start:start + chunk_size])
                for start in range(0, n, chunk_size)
            ]
            return tf.concat(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        # Graph path.
        num_positions_t = tf.shape(moves_21)[0]

        def _body(start, embeddings):
            end = tf.minimum(start + chunk_size, num_positions_t)
            embeddings = embeddings.write(start // chunk_size, _project_chunk(moves_21[start:end]))
            return end, embeddings

        num_chunks = (num_positions_t + chunk_size - 1) // chunk_size
        embeddings_ta = tf.TensorArray(
            dtype=tf.float32,
            size=num_chunks,
            dynamic_size=False,
            element_shape=[None, self.hidden_dim],
        )
        _, embeddings_ta = tf.while_loop(
            cond=lambda start, _: start < num_positions_t,
            body=_body,
            loop_vars=(tf.constant(0), embeddings_ta),
            parallel_iterations=1,
        )
        return embeddings_ta.concat()

    def call(self, inputs, training=None, mask=None):
        move_features = inputs  # (batch, num_moves, 21, 8, 8)
        batch_size = tf.shape(move_features)[0]
        num_moves = tf.shape(move_features)[1]

        moves_reshaped = tf.reshape(move_features, [-1, 21, 8, 8])  # (batch*num_moves, 21, 8, 8)
        move_embeddings = self._project_moves(moves_reshaped, training=training)
        x = tf.reshape(move_embeddings, [batch_size, num_moves, self.hidden_dim])

        x_even = x[:, 0::2, :]
        x_odd = x[:, 1::2, :]
        pair_count = tf.minimum(tf.shape(x_even)[1], tf.shape(x_odd)[1])
        x_even = x_even[:, :pair_count, :]
        x_odd = x_odd[:, :pair_count, :]
        x = tf.concat([x_even, x_odd], axis=-1)
        x = self.pair_projection(x)

        pair_mask = None
        if mask is not None:
            mask_even = tf.cast(mask[:, 0::2], dtype=x.dtype)
            mask_odd = tf.cast(mask[:, 1::2], dtype=x.dtype)
            mask_even = mask_even[:, :pair_count]
            mask_odd = mask_odd[:, :pair_count]
            pair_mask = tf.minimum(mask_even, mask_odd)

        positions = self.positional_encoding[:pair_count, :]
        x = x + positions

        x = tf.expand_dims(x, axis=2)  # (batch, num_pairs, 1, hidden_dim)
        x = self.vit(x, training=training, mask=pair_mask)
        x = tf.squeeze(x['pre_logits'], axis=2)  # (batch, num_pairs, hidden_dim)

        if pair_mask is not None:
            mask_expanded = tf.expand_dims(tf.cast(pair_mask, dtype=x.dtype), axis=-1)
            x_masked = x * mask_expanded
            aggregated = tf.math.divide_no_nan(
                tf.reduce_sum(x_masked, axis=1),
                tf.reduce_sum(mask_expanded, axis=1)
            )
        else:
            aggregated = tf.reduce_mean(x, axis=1)
        return aggregated
    
    def model(self):
        x = tf.keras.Input(shape=(self.max_moves, self.move_feature_dim))
        return tf.keras.Model(inputs=x, outputs=self.call(x))
    
if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = GameAggregateViT(move_feature_dim=21*8*8)
    # model.build(input_shape=(None, 500, 21, 8, 8))
    # model.summary()