import yaml
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers # type: ignore
import tensorflow_models as tfm # type: ignore

from tfprocess import TFProcess
from stylometry.ViTOneHot.game_aggregate_vit import GameAggregateViT

CONFIG_FILE_PATH = 'configs/744706.yaml'

class WinRateStyloModel(tf.keras.Model):
  def __init__(
      self,
      style_vec_size,
      num_hidden_layers=4,
      hidden_dim=768,
      **kwargs
  ):
    super(WinRateStyloModel, self).__init__(**kwargs)

    self.style_vec_size = style_vec_size
    self.num_hidden_layers = num_hidden_layers
    self.hidden_dim = hidden_dim

    with open(CONFIG_FILE_PATH, 'r') as file:
        cfg = yaml.safe_load(file)
    
    tfp = TFProcess(cfg)
    tfp.init_net(use_heads=False)
    tfp.restore()

    self.conv = tfp.model
    # self.conv.trainable = False
    
    self.ffn = tf.keras.models.Sequential([tf.keras.layers.Dense(units=hidden_dim, activation='relu') for _ in range(num_hidden_layers)])

    self.wdl_head = tf.keras.layers.Dense(units=3, activation=None)
  
  def call(self, inputs, training=None, mask=None):
    styles_and_move_features = inputs  # (batch, style_vec_size*2+112*8*8)
    move_features = styles_and_move_features[:, self.style_vec_size * 2:]  # (batch, 112*8*8)
    conv_out = self.conv(tf.reshape(move_features, (-1, 112, 8, 8)), training=training)  # (batch, hidden_dim)
    styles_and_move_features = tf.concat([tf.cast(styles_and_move_features[:, :self.style_vec_size * 2], tf.float32), tf.reshape(conv_out, (-1, 128*8*8))], axis=-1)  # (batch, style_vec_size*2 + hidden_dim)
    ffn_out = self.ffn(styles_and_move_features, training=training)  # (batch, hidden_dim)
    wdl_out = self.wdl_head(ffn_out, training=training)  # (batch, 3)
    return wdl_out
  
  def model(self):
    x = tf.keras.Input(shape=(self.style_vec_size * 2 + 112 * 8 * 8,))
    return tf.keras.Model(inputs=x, outputs=self.call(x))
    
if __name__ == "__main__":
  pass
  # gpus = tf.config.experimental.list_physical_devices('GPU')
  # for gpu in gpus:
  #     tf.config.experimental.set_memory_growth(gpu, True)
  # model = WinRateStyloModel(move_feature_dim=112*8*8)
  # model.build(input_shape=(None, 500, 112, 8, 8))
  # model.summary()