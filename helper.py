
import numpy as np 
import tensorflow as tf
#@title
def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates

#@title
def positional_encoding(position, d_model):
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    del angle_rads

    return tf.cast(pos_encoding, dtype=tf.float32)




#@title
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    #matmul_qk = sliding_chunks_matmul_qk(q, k , 7)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(attention_weights, v)
    del matmul_qk
    del scaled_attention_logits
    return output, attention_weights

#@title
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

def beam_search_decoder(predictions, top_k = 3):
    #start with an empty sequence with zero score
    output_sequences = [([], 0)]
    
    #looping through all the predictions
    for token_probs in predictions:
        new_sequences = []
        
        #append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                #considering log-likelihood for scoring
                new_score = old_score + tf.math.log(token_probs[char_index])
                new_sequences.append((new_seq, new_score))
                
        #sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
        
        #select top-k based on score 
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]
        
    return output_sequences
    
#@title
def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    final_loss = tf.reduce_sum(loss_)/tf.reduce_sum(mask)
    del loss_
    del mask
    return final_loss

def evaluate(input_document, tokenizer, encoder_maxlen, decoder_maxlen, transformer):
    input_document = '[CLS]' + input_document + '[SEP]'
    input_document = [tokenizer(d,return_tensors='tf').data['input_ids'].numpy().tolist()[0] for d in input_document]
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen, padding='post', truncating='post')
    input_document = tf.cast(input_document, dtype=tf.int32)
    encoder_input = tf.expand_dims(input_document[0], 0)
    decoder_input = tokenizer("<go>",return_tensors='tf').data['input_ids'].numpy().tolist()[0]
    output = tf.expand_dims(decoder_input, 0)
    
    get_final_value = tf.zeros((1,1,64000))
    for i in range(decoder_maxlen):
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
        get_final_value = np.vstack([get_final_value, predictions])
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == tokenizer("stop",return_tensors='tf').data['input_ids'].numpy().tolist()[0][1]:
            return tf.squeeze(output, axis=0),tf.convert_to_tensor(get_final_value) , attention_weights 

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0),tf.convert_to_tensor(get_final_value), attention_weights 

def beam_search_decoder(predictions, top_k = 3):
    #start with an empty sequence with zero score
    output_sequences = [([], 0)]
    
    #looping through all the predictions
    for token_probs in predictions:
        new_sequences = []
        
        #append new tokens to old sequences and re-score
        for old_seq, old_score in output_sequences:
            for char_index in range(len(token_probs)):
                new_seq = old_seq + [char_index]
                #considering log-likelihood for scoring
                new_score = old_score + tf.math.log(token_probs[char_index])
                new_sequences.append((new_seq, new_score))
                
        #sort all new sequences in the de-creasing order of their score
        output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)
        
        #select top-k based on score 
        # *Note- best sequence is with the highest score
        output_sequences = output_sequences[:top_k]
        
    return output_sequences

#@title
def summarize(input_document):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    result = evaluate(input_document=input_document)
    summarized = result[0].numpy()
    result_beam = beam_search_decoder(result[1])
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summarized, result_beam # since there is just one translated document
    
#@title
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

#@title
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    del look_ahead_mask
    del dec_target_padding_mask
    return enc_padding_mask, combined_mask, dec_padding_mask
