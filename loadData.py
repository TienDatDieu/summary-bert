import joblib
import tensorflow as tf 
import re
import pandas as pd
from config import *

def read_data(tokenizer, filetrain='train_pharagraph.jl', filetarget = 'target_strings.jl'):
    train_pharagraph = joblib.load(filetrain)
    target_strings = joblib.load(filetarget)

    targets = []
    for ele in target_strings:
        ele = re.sub(r'[\W_]', ' ', ele)
        ele = re.sub(r'[^\w\s]', '', ele)
        ele = re.sub(r'\t\n', '', ele)
        targets.append(ele)

    training_input = []
    for ele in train_pharagraph:
        ele = re.sub(r'[\W_]', ' ', ele)
        ele = re.sub(r'[^\w\s]', '', ele)
        ele = re.sub(r'\t\n', '', ele)
        training_input.append(ele)

    document = pd.Series(training_input)
    summary = pd.Series(targets)

    doc_list = []
    sum_list = []
    for index,doc in enumerate(document):
        if len(doc) < 2000:
            doc_list.append(doc)
            sum_list.append(summary[index])

    document = pd.Series(doc_list)
    summary = pd.Series(sum_list)

    # document = pd.Series(document)
    # summary = pd.Series(summary)

    summary = summary.apply(lambda x: tokenizer.special_tokens_map['cls_token'] + x + tokenizer.special_tokens_map['sep_token'])
    document = document.apply(lambda x: tokenizer.special_tokens_map['cls_token'] + x + tokenizer.special_tokens_map['sep_token'])

    document_bert = [tokenizer(d,return_tensors='tf').data['input_ids'].numpy().tolist()[0] for d in document]
    summary_bert = [tokenizer(d,return_tensors='tf').data['input_ids'].numpy().tolist()[0] for d in summary]
    
    inputs_bert = tf.keras.preprocessing.sequence.pad_sequences(document_bert, maxlen=encoder_maxlen, padding='post', truncating='post')
    targets_bert = tf.keras.preprocessing.sequence.pad_sequences(summary_bert, maxlen=decoder_maxlen, padding='post', truncating='post')


    inputs = tf.cast(inputs_bert, dtype=tf.int32)
    targets = tf.cast(targets_bert, dtype=tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((inputs[:int(len(inputs)*0.9)], targets[:int(len(inputs)*0.9)])).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return dataset, document[int(len(inputs)*0.9):], summary[int(len(inputs)*9):]