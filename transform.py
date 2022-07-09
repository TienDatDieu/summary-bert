from encoder import Encoder
from decoder import Decoder
import tensorflow as tf
from log_manager import logger


#@title

class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, word2Topic, list_topic_count, rate=0.1):
        super(TransformerModel, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

        self.word2Topic = word2Topic
        self.list_topic_count = list_topic_count
    
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask, alpha = 0):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        # full_topic = []
        # a_topic = []
        # a = []
        # for e in inp.numpy():
        #   a = 18*[0]
        #   for el in e:
        #     a = a + self.word2Topic[int(el)].numpy()
        #   a_topic.append(a/300)
        # full_topic.append(a_topic)
       
        # topic_arg1 = tf.matmul(tf.tile(tf.expand_dims(self.word2Topic, 0),[final_output.shape[0], 1, 1]), tf.cast(tf.reshape(full_topic,(final_output.shape[0],18,1)), dtype=tf.float32))
        # topic_arg2 = tf.tile(tf.reshape(topic_arg1, (topic_arg1.shape[0], 1 , topic_arg1.shape[1])), [1,final_output.shape[1],1])
        
        # topic_arg3 =  tf.truediv(topic_arg2 , tf.reduce_max(topic_arg2))  
        # sum_all = final_output * (1-alpha) + alpha*tf.cast(topic_arg3, dtype=tf.float32)
        return final_output, enc_output, attention_weights
