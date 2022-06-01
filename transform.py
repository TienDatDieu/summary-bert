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
        logger.info("Before Encoder and Decoder")
        print("Before Encoder and Decoder")
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        full_topic = []
        a_topic = []
        a = []
        for e in inp.numpy():
          a = 18*[0]
          for el in e:
            a = a + self.word2Topic[int(el)].numpy()
          a_topic.append(a/300)
        full_topic.append(a_topic)
        # print(final_output.dtype.as_numpy_dtype)
        # print(self.word2Topic.dtype.as_numpy_dtype)
        # print(tf.reshape(full_topic,(final_output.shape[0],18,1)).dtype.as_numpy_dtype)
        # print(tf.tile(tf.expand_dims(self.word2Topic, 0),[final_output.shape[0], 1, 1]).dtype.as_numpy_dtype)
        topic_arg1 = tf.matmul(tf.tile(tf.expand_dims(self.word2Topic, 0),[final_output.shape[0], 1, 1]), tf.cast(tf.reshape(full_topic,(final_output.shape[0],18,1)), dtype=tf.float32))
        # print("topic_arg1",final_output)
        topic_arg2 = tf.tile(tf.reshape(topic_arg1, (topic_arg1.shape[0], 1 , topic_arg1.shape[1])), [1,final_output.shape[1],1])
        # print("topic_arg2", topic_arg2)
        # topic_arg = tf.matmul(self.word2Topic, tf.reshape(tf.convert_to_tensor([number / sum(self.list_topic_count) for number in self.list_topic_count], dtype = tf.float64),(18,1)))
        # #word2Topic is p(word|Topic)
        # print("topic_arg", tf.reshape(tf.convert_to_tensor([number / sum(self.list_topic_count) for number in self.list_topic_count], dtype = tf.float64),(18,1)))
        # topic_arg_reshape = tf.transpose(topic_arg)
        # print("finalouput shape",final_output.shape)
        # topic_mul = tf.tile(topic_arg_reshape, multiples = [final_output.shape[1], 1])
        # print("topic_mul",topic_mul)
        # topic_mul_full = tf.tile(tf.expand_dims(topic_mul,0), multiples = [final_output.shape[0], 1, 1])
        
        # topic_arg3 = tf.truediv(tf.subtract(topic_arg2, tf.reduce_min(topic_arg2)), tf.subtract(tf.reduce_max(topic_arg2), tf.reduce_min(topic_arg2))) 
        topic_arg3 =  tf.truediv(topic_arg2 , tf.reduce_max(topic_arg2))  
        sum_all = final_output * (1-alpha) + alpha*tf.cast(topic_arg3, dtype=tf.float32)
        return sum_all, enc_output, attention_weights
