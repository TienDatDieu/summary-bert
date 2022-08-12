# coding=utf-8
import tensorflow as tf
from CustomSchedule import CustomSchedule
from loadData import read_data 
from config import *
import joblib
import time
from transform import TransformerModel 
from helper import *

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
from transformers import TFBertModel
model = TFBertModel.from_pretrained("bert-base-multilingual-cased")
tf.config.run_functions_eagerly(True)

train_loss = tf.keras.metrics.Mean(name='train_loss')

from queue import PriorityQueue
import operator
from log_manager import logger
gpus = tf.config.list_physical_devices('GPU')
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)




class BeamSearchNode(object):
	def __init__(self, prev_node, token_id, log_prob):
		self.finished = False   # Determine if the hypothesis decoding is finished
		self.prev_node = prev_node
		self.token_id = token_id
		self.log_prob = log_prob

		if prev_node is None:
			self.seq_tokens = [token_id]
		else:
			self.seq_tokens = prev_node.seq_tokens + [token_id]

		self.seq_len = len(self.seq_tokens)

		if token_id == tokenizer.eos_token_id:
			self.finished = True

	def eval(self):
		alpha = 1.0
		reward = 0

		# Add here a function for shaping a reward
		return self.log_prob / float(self.seq_len - 1 + 1e-6) + alpha * reward

	def __lt__(self, other):
		return self.seq_len < other.seq_len

	def __gt__(self, other):
		return self.seq_len > other.seq_len

def evaluate_beam(input_document, n_best, k_beam, transformer):
    # input_document = '[CLS]' + input_document + '[SEP]'
    input_document = [tokenizer(d,return_tensors='tf').data['input_ids'].numpy().tolist()[0] for d in input_document]
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')
    input_document = tf.cast(input_document, dtype=tf.int32)
    encoder_input = tf.expand_dims(input_document[0], 0)
    decoder_input = tokenizer(tokenizer.special_tokens_map['cls_token'],return_tensors='tf').data['input_ids'].numpy().tolist()[0]
    output = tf.expand_dims(decoder_input, 0)
    decoded_batch = []
    # khởi tạo mảng hypothesis
    beam_hypotheses = []
    # khởi tạo nút root (token là begin)
    start_node = BeamSearchNode(prev_node=None, token_id=tokenizer.special_tokens_map['cls_token'], log_prob=0)
    # add nút root vào hypothesis (add Tuple, điểm của nó và chính nó)
    beam_hypotheses.append((-start_node.eval(), start_node))
    end_nodes = []

    for i in range(decoder_maxlen):
      # tạo mảng candidate có kiểu Priority Queue
      candidates = PriorityQueue()
      # vòng lặp for cho các tuple trong hypothesis (score, node)
      for score, node in beam_hypotheses:
        dec_seq = node.seq_tokens
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        predictions, attention_weights,_ = transformer(
            encoder_input, 
            output,
            False,
            enc_padding_mask,
            combined_mask,
            dec_padding_mask
        )
        predictions = predictions[: ,-1:, :]
        sorted_probs = tf.sort(predictions[0][0], axis=-1, direction = 'DESCENDING') 
        sorted_indices = tf.argsort(predictions[0][0], axis=-1, direction = 'DESCENDING')
        # lấy ra prob từ prediction
        # sort prob đó giảm dần 
        for i in range(n_best):
            decoded_token = sorted_indices[i]
            log_prob = sorted_probs[i]
    
            next_node = BeamSearchNode(prev_node=node, token_id=decoded_token, log_prob=node.log_prob + log_prob)
            
            if decoded_token == tokenizer.eos_token_id:
                end_nodes.append((-next_node.eval(), next_node))
            else:
                candidates.put((-next_node.eval(), next_node))
        # lấy ra beam size các phần tử có prob cao nhất (danh sách các candidate)
        # khởi tạo các node vào mảng candidate (prev_node = current vòng lặp)
        # append vào mảng hypothesis
        if len(end_nodes) >= k_beam:
            break
        beam_hypotheses = [candidates.get() for _ in range(k_beam)]
      # Get beamsize các phần tử ở Queue
      # Gán lại cho hypothesis
    best_hypotheses = []
    sorted_beam_hypotheses = sorted(beam_hypotheses, key=operator.itemgetter(0))
    
    if len(end_nodes) < k_beam:
        for i in range(k_beam - len(end_nodes)):
            end_nodes.append(sorted_beam_hypotheses[i])
    sorted_end_nodes = sorted(end_nodes, key=operator.itemgetter(0))

    for i in range(k_beam):
        score, end_node = sorted_end_nodes[i]
        best_hypotheses.append((score, end_node.seq_tokens))
    decoded_batch.append(best_hypotheses)
    # best hypothesis
    # lấy
    return decoded_batch

#@title
@tf.function
def train_step(inp, tar, transformer):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, enc_output, att_weights = transformer(
            inp, tar_inp, 
            True, 
            enc_padding_mask, 
            combined_mask, 
            dec_padding_mask,
        )
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    del inp 
    del tar
    del tar_inp
    del tar_real
    del loss
    del enc_padding_mask
    del combined_mask
    del dec_padding_mask
    del att_weights
    del gradients
    return 


def train(transformer):

    dataset, val_input, val_output = read_data(tokenizer)

    enc_out = []
    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch))
        start = time.time()
        train_loss.reset_states()
        for (batch, (inp, tar)) in enumerate(dataset):
            train_step(inp, tar, transformer)
            if batch > 0 and (batch) % 100 == 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
                logger.info('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

        
        logger.info('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
        logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    return val_input, val_output

if __name__ == "__main__":
    word2Topic = joblib.load('word2Topic.jl')
    list_topic_count = joblib.load('list_topic_count.jl')
    checkpoint_path = "checkpoints"
    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer = TransformerModel(
        num_layers, 
        d_model, 
        num_heads, 
        dff,
        encoder_vocab_size, 
        decoder_vocab_size, 
        pe_input=encoder_vocab_size, 
        pe_target=decoder_vocab_size,
        word2Topic=word2Topic,
        list_topic_count=list_topic_count
        ) 
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    
    val_input, val_output = train(transformer)

    for input_document in val_input:
        result_beam = evaluate_beam(input_document, 3, 3, transformer)
        
        for e in result_beam:
            sentence_result = []
            for i in e:
                for s in i[1]:
                    if isinstance(s, str):
                        sentence_result.append(s)
                    else:
                        sentence_result.append(tokenizer.decode(s.numpy()))
            print(sentence_result)
            logger.info(sentence_result)
    
    


    
