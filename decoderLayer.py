import tensorflow as tf
from multiheadAttention import MultiHeadAttention
from helper import *
#@title
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        dec_output, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        dec_output = self.dropout1(dec_output, training=training)
        out1 = self.layernorm1(dec_output + x)

        dec_output, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        dec_output = self.dropout2(dec_output, training=training)
        out2 = self.layernorm2(dec_output + out1)

        dec_output = self.ffn(out2) 
        out3 = self.dropout3(dec_output, training=training)
        final_out = self.layernorm3(out3 + out2)

        return final_out, attn_weights_block1, attn_weights_block2
