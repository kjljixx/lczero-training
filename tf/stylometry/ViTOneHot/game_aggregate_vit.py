import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers # type: ignore
from official.vision.modeling.backbones.vit import VisionTransformer  # type: ignore

from tfprocess import TFProcess

CONFIG_FILE_PATH = 'configs/256x10-t1.yaml'

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

        # Build the frozen LC0 body under mixed_bfloat16 so matmuls run in bf16
        # (weights stay fp32 for checkpoint restore). Trainable ViT/heads below
        # stay on the default float32 policy.
        _prev_policy = tf.keras.mixed_precision.global_policy()
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
        try:
            tfp = TFProcess(cfg)
            tfp.init_net(use_heads=False)
            tfp.restore()

            # ONLY extract body, not heads. Freeze LC0 weights so stylometry
            # training only updates pair_projection / ViT / elo heads (avoids
            # GradientTape OOM from encoder activations).
            input_var = tf.keras.Input(shape=(112, 8, 8))
            assert isinstance(cfg, dict)
            self.move_compress = None
            if cfg['model'].get('encoder_layers', 0) > 0:
                embedding_size = cfg['model'].get('embedding_size', cfg['model'].get('filters'))
                flow, _ = tfp.create_encoder_body(input_var, embedding_size)
                flow = tf.keras.layers.GlobalAveragePooling1D()(flow)
                self.move_projection = tf.keras.Model(inputs=input_var, outputs=flow)
                self.lc0_embedding_dim = int(embedding_size)
            else:
                filters = cfg['model']['filters']
                self.move_projection = tfp.model
                self.lc0_embedding_dim = filters * 8 * 8
        finally:
            tf.keras.mixed_precision.set_global_policy(_prev_policy)

        if self.move_compress is None and self.lc0_embedding_dim != self.hidden_dim:
            if cfg['model'].get('encoder_layers', 0) > 0:
                self.move_compress = layers.Dense(units=self.hidden_dim, activation='relu')
            else:
                print("Stylo ViT Using compression layer")
                self.move_compress = tf.keras.Sequential([
                    layers.Flatten(),
                    layers.Dense(units=self.hidden_dim, activation='relu'),
                ])

        self.move_projection.trainable = False
        print("Frozen LC0 move_projection body (bf16 compute; stylometry trains ViT + heads only)")
        self._ensure_lc0_bf16_policy()

        self.pair_projection = layers.Dense(units=hidden_dim, activation='relu')

        # BiasAdd GPU kernels multiply N*H*W*C as int32. For emb=256 that
        # allows N < ~131k; for emb=768 the limit is ~43k. Encoder activation
        # memory is the practical bound — 4096 is safe for 256x10 and cuts
        # while_loop / launch overhead vs LC0's train microbatch of 512.
        # Fixed size keeps XLA from retracing on every masked remainder length.
        self.move_proj_chunk_size = 4096
        self._project_chunk = None  # built below / lazily for checkpoint loads
        self._get_project_chunk()

        #sin position encoding like paper recommended
        self.positional_encoding = get_sinusoidal_positional_encoding(
            max_positions=self.paired_max_moves,
            d_model=hidden_dim
        )
        
        # Create input specs for VisionTransformer
        # VisionTransformer expects (batch, height, width, channels) input in 4D
        # Use paired_max_moves as the height dimension.
        input_specs = tf.keras.layers.InputSpec(shape=[None, self.paired_max_moves, 1, hidden_dim])
        
        self.vit = VisionTransformer(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_dim,
            mlp_dim=mlp_dim,
            input_specs=input_specs,
            patch_size=1,  # Since we already have embeddings, no patching needed
        )

    def _ensure_lc0_bf16_policy(self):
        """Force mixed_bfloat16 compute on the frozen LC0 body (weights stay fp32).

        Functional models expose a read-only dtype_policy, so only leaf layers
        that accept assignment are updated.
        """
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')

        def _apply(layer):
            sublayers = getattr(layer, 'layers', None)
            if sublayers:
                for sub in sublayers:
                    _apply(sub)
            try:
                layer.dtype_policy = policy
            except AttributeError:
                # Functional / nested Model: read-only @property — children cover it.
                pass

        _apply(self.move_projection)

    def _get_project_chunk(self):
        """XLA kernel for one fixed-size LC0 chunk; created once per instance."""
        if getattr(self, '_project_chunk', None) is None:
            chunk_size = int(getattr(self, 'move_proj_chunk_size', 4096))
            # Upgrade older checkpoints that still carry chunk_size=512.
            if chunk_size < 4096:
                chunk_size = 4096
                self.move_proj_chunk_size = chunk_size
            self._ensure_lc0_bf16_policy()
            self._project_chunk = self._make_project_chunk_fn(chunk_size)
        return self._project_chunk

    def _make_project_chunk_fn(self, chunk_size):
        """Build an XLA-compiled LC0 forward that always sees `chunk_size` rows."""
        move_projection = self.move_projection
        out_dim = self.lc0_embedding_dim

        @tf.function(
            jit_compile=True,
            input_signature=[tf.TensorSpec(shape=[chunk_size, 21, 8, 8], dtype=tf.float32)],
        )
        def _project_chunk(chunk_21):
            chunk_bf16 = tf.cast(chunk_21, tf.bfloat16)
            padded = tf.concat([
                chunk_bf16[:, :13],
                tf.zeros((chunk_size, 91, 8, 8), dtype=tf.bfloat16),
                chunk_bf16[:, -8:],
            ], axis=1)
            out = move_projection(padded, training=False)
            out = tf.cast(out, tf.float32)
            out = tf.stop_gradient(out)
            return tf.reshape(out, [chunk_size, out_dim])

        return _project_chunk

    def _project_moves_lc0(self, moves_21):
        """Pad to 112 planes and run frozen LC0 body in chunks.

        Returns stop-gradient embeddings of shape (N, lc0_embedding_dim).
        Callers that need grads through move_compress must apply it outside
        GradientTape.stop_recording().

        Chunking bounds:
        1) BiasAdd GPU kernels multiply N*H*W*C as int32 (overflow at ~131k
           positions for emb=256, ~43k for emb=768).
        2) Encoder activation memory; 4096 is a safe tradeoff for 256x10.
        """
        chunk_size = int(getattr(self, 'move_proj_chunk_size', 4096))
        if chunk_size < 4096:
            chunk_size = 4096
            self.move_proj_chunk_size = chunk_size
        project_chunk = self._get_project_chunk()
        num_positions_t = tf.shape(moves_21)[0]
        out_dim = self.lc0_embedding_dim

        def _run_chunk(chunk):
            n = tf.shape(chunk)[0]
            # Pad to fixed chunk_size so the XLA binary stays monomorphic.
            chunk = tf.pad(chunk, [[0, chunk_size - n], [0, 0], [0, 0], [0, 0]])
            return project_chunk(chunk)[:n]

        def _empty():
            return tf.zeros((0, out_dim), dtype=tf.float32)

        def _chunked():
            # Prefer while_loop under tf.function so graph mode does not unroll
            # tens of thousands of positions into a giant Python list of ops.
            if tf.executing_eagerly():
                n = int(moves_21.shape[0]) if moves_21.shape[0] is not None else None
                if n is not None:
                    chunks = [
                        _run_chunk(moves_21[start:start + chunk_size])
                        for start in range(0, n, chunk_size)
                    ]
                    return tf.concat(chunks, axis=0) if len(chunks) > 1 else chunks[0]

            def _body(start, embeddings):
                end = tf.minimum(start + chunk_size, num_positions_t)
                embeddings = embeddings.write(
                    start // chunk_size, _run_chunk(moves_21[start:end])
                )
                return end, embeddings

            num_chunks = (num_positions_t + chunk_size - 1) // chunk_size
            embeddings_ta = tf.TensorArray(
                dtype=tf.float32,
                size=num_chunks,
                dynamic_size=False,
                element_shape=[None, out_dim],
            )
            _, embeddings_ta = tf.while_loop(
                cond=lambda start, _: start < num_positions_t,
                body=_body,
                loop_vars=(tf.constant(0), embeddings_ta),
                parallel_iterations=1,
            )
            return embeddings_ta.concat()

        return tf.cond(tf.equal(num_positions_t, 0), _empty, _chunked)

    def lc0_encode(self, move_features, mask=None):
        """Frozen LC0 encode: (batch, moves, 21, 8, 8) -> (batch, moves, lc0_dim).

        If mask is provided (batch, moves), only positions with mask > 0 are
        run through the LC0 body; padded slots get zero embeddings.
        """
        batch_size = tf.shape(move_features)[0]
        num_moves = tf.shape(move_features)[1]
        moves_reshaped = tf.reshape(move_features, [-1, 21, 8, 8])
        out_dim = self.lc0_embedding_dim
        flat_count = batch_size * num_moves

        if mask is not None:
            flat_mask = tf.reshape(mask, [-1]) > 0
            valid_moves = tf.boolean_mask(moves_reshaped, flat_mask)
            valid_embeddings = self._project_moves_lc0(valid_moves)
            indices = tf.where(flat_mask)
            full = tf.zeros(tf.stack([flat_count, out_dim]), dtype=tf.float32)
            full = tf.tensor_scatter_nd_update(full, indices, valid_embeddings)
            return tf.reshape(full, [batch_size, num_moves, out_dim])

        move_embeddings = self._project_moves_lc0(moves_reshaped)
        return tf.reshape(move_embeddings, [batch_size, num_moves, out_dim])

    def from_lc0_embeddings(self, lc0_embeddings, training=None, mask=None):
        """Trainable path from LC0 embeddings: compress + pair ViT + pool."""
        x = lc0_embeddings
        if self.move_compress is not None:
            x = self.move_compress(x, training=training)
        return self._aggregate_from_move_embeddings(x, training=training, mask=mask)

    def _aggregate_from_move_embeddings(self, x, training=None, mask=None):
        """x: (batch, num_moves, hidden_dim) -> (batch, hidden_dim)."""
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

    def call(self, inputs, training=None, mask=None):
        move_features = inputs  # (batch, num_moves, 21, 8, 8)
        lc0_embeddings = self.lc0_encode(move_features, mask=mask)
        return self.from_lc0_embeddings(lc0_embeddings, training=training, mask=mask)
    
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
