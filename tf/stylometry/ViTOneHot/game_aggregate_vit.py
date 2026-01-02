import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers # type: ignore
import tensorflow_models as tfm # type: ignore

from tfprocess import TFProcess

CONFIG_FILE_PATH = 'configs/744706.yaml'

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

        with open(CONFIG_FILE_PATH, 'r') as file:
            cfg = yaml.safe_load(file)
        
        tfp = TFProcess(cfg)
        tfp.init_net(use_heads=False)
        tfp.restore()
        
        # ONLY extract body, not heads
        input_var = tf.keras.Input(shape=(21, 8, 8))
        assert isinstance(cfg, dict)
        if cfg['model'].get('encoder_layers', 0) > 0:
            assert False, "not implemented"
            flow, _ = tfp.create_encoder_body(input_var, cfg['model']['filters'])
            print(flow.shape)
            assert flow.shape[2] == self.hidden_dim, f"Expected encoder output dim {self.hidden_dim}, got {flow.shape[2]}"
            self.move_projection = tf.keras.Model(inputs=input_var, outputs=flow)
        else:
            filters = cfg['model']['filters']
            # assert self.hidden_dim == filters * 8 * 8, f"Expected hidden dim {filters * 8 * 8}, got {self.hidden_dim}"
            self.move_projection = tfp.model
            if filters * 8 * 8 != self.hidden_dim:
              print("Stylo ViT Using compression layer")
              self.move_projection = tf.keras.Sequential([
                  self.move_projection,
                  layers.Flatten(),
                  layers.Dense(units=self.hidden_dim, activation='relu')
              ])
        
        #sin position encoding like paper recommended
        self.positional_encoding = get_sinusoidal_positional_encoding(
            max_positions=max_moves,
            d_model=hidden_dim
        )
        
        # Create input specs for VisionTransformer
        # VisionTransformer expects (batch, height, width, channels) input in 4D
        # Use max_moves as the height dimension (must be concrete, not None)
        input_specs = tf.keras.layers.InputSpec(shape=[None, max_moves, 1, hidden_dim])
        
        self.vit = tfm.vision.backbones.VisionTransformer(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_dim,
            mlp_dim=mlp_dim,
            input_specs=input_specs,
            patch_size=1,  # Since we already have embeddings, no patching needed
        )
    
    def call(self, inputs, training=None, mask=None):
      move_features = inputs  # (batch, num_moves, 21, 8, 8)
      batch_size = tf.shape(move_features)[0]
      num_moves = tf.shape(move_features)[1]
      
      moves_reshaped = tf.reshape(move_features, [-1, 21, 8, 8])  # (batch*num_moves, 21, 8, 8)
      moves_reshaped = tf.concat([moves_reshaped[:, :13], tf.zeros((batch_size*num_moves, 91, 8, 8), dtype=int8), moves_reshaped[:, -8:]], axis=1) # (batch*num_moves, 112, 8, 8)
      move_embeddings = self.move_projection(moves_reshaped, training=training)  # (batch*num_moves, filters, 8, 8)

      move_embeddings = tf.reshape(move_embeddings, [-1, self.hidden_dim]) # (batch*num_moves, hidden_dim)

      x = tf.reshape(move_embeddings, [batch_size, num_moves, self.hidden_dim]) # (batch, num_moves, hidden_dim)

      positions = self.positional_encoding[:num_moves, :]
      x = x + positions
      
      # Reshape to 4D for VisionTransformer: (batch, num_moves, 1, hidden_dim)
      x = tf.expand_dims(x, axis=2)  # (batch, num_moves, 1, hidden_dim)
      x = self.vit(x, training=training, mask=mask)
      
      # Reshape back to 3D: (batch, num_moves, hidden_dim)
      x = tf.squeeze(x['pre_logits'], axis=2)  # (batch, num_moves, hidden_dim)
      
      if mask is not None:
          mask_expanded = tf.expand_dims(tf.cast(mask, dtype=x.dtype), axis=-1)
          x_masked = x * mask_expanded
          aggregated = tf.math.divide_no_nan(tf.reduce_sum(x_masked, axis=1), tf.reduce_sum(mask_expanded, axis=1))
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