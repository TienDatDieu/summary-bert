#@title
# hyper-params
num_layers = 4
d_model = 512
dff = 512
num_heads = 8
EPOCHS = 10000

encoder_vocab_size = 119547 # tokenizer.vocab_size
decoder_vocab_size = 119547 # tokenizer.vocab_size

encoder_maxlen = 512
decoder_maxlen = 200

BUFFER_SIZE = 20000
BATCH_SIZE = 4