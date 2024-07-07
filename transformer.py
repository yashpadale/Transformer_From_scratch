import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import string
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize






def number_to_words(predictions, dictionary):
    # Invert the dictionary
    inverted_dictionary = {v: k for k, v in dictionary.items()}

    predicted_sentences = []

    for prediction_row in predictions:
        words_row = []
        for index in prediction_row:
            # Check if the index exists in the inverted dictionary
            word = inverted_dictionary.get(index)
            if word is not None:
                words_row.append(word)
        predicted_sentence = ' '.join(words_row)
        predicted_sentences.append(predicted_sentence)

    return predicted_sentences



def split_list(lst, percentage: float):
    len_75 = int(len(lst) * percentage)
    first_list = lst[:len_75]
    second_list = lst[len_75:]
    return first_list, second_list


def rearrange(batches):
    X = []
    Y = []
    for i in range(len(batches) - 1):
        X.append(batches[i])
        Y.append(batches[i + 1])
    return X, Y


def return_dict(unique_words:list):
    dictionary={}
    for i in range(len(unique_words)):
        dictionary[unique_words[i]]=i+1
    return dictionary

from nltk.tokenize import word_tokenize

def pad_segments(content: str, maxlen: int):
    maxlen = maxlen - 1  # Adjust maxlen to account for the <start> and <end> tokens
    segments = content.split('<end>')
    padded_segments = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Ensure <start> and <end> are treated as single tokens
        segment = segment.replace('<start>', 'STARTTOKENPLACEHOLDER')
        segment = segment.replace('<end>', 'ENDTOKENPLACEHOLDER')

        # Tokenize the segment
        tokens = word_tokenize(segment)

        # Replace the placeholders with original tokens
        tokens = ['<start>' if token == 'STARTTOKENPLACEHOLDER' else token for token in tokens]
        tokens = ['<end>' if token == 'ENDTOKENPLACEHOLDER' else token for token in tokens]

        # Ensure the segment starts with '<start>'
        if tokens[0] != '<start>':
            print(tokens)
            print(segment)
            raise ValueError("Segment does not start with '<start>'")

        # Remove '<start>' token for padding calculation
        start_token = tokens.pop(0)

        # Check if the segment exceeds maxlen - 1 (considering <start> and <end>)
        if len(tokens) > maxlen - 1:
            tokens = tokens[:maxlen - 1]

        # Add '<space>' tokens if necessary to pad the segment to maxlen - 1
        padding_needed = maxlen - len(tokens) - 1
        tokens.extend(['<space>'] * padding_needed)

        # Reinsert '<start>' token at the beginning
        tokens.insert(0, start_token)

        # Add '<end>' token at the end
        tokens.append('<end>')

        padded_segments.append(' '.join(tokens))

    # Join all padded segments into the final output
    final_output = ' '.join(padded_segments)
    return final_output

def read_file(filename):
    with open(filename, 'r',encoding='utf-8') as file:
        content = file.read()
    return content


def split_and_sort(string):
    string = string.replace('<start>', 'STARTTOKENPLACEHOLDER')
    string = string.replace('<end>', 'ENDTOKENPLACEHOLDER')
    string = string.replace('<space>', 'SPACETOKENPLACEHOLDER')
    tokens = word_tokenize(string)
    tokens = ['<start>' if token == 'STARTTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<end>' if token == 'ENDTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<space>' if token == 'SPACETOKENPLACEHOLDER' else token for token in tokens]
    unique_words_list = list(set(tokens))
    return unique_words_list


def return_order(dict_, content: str):
    # Replace special tokens with placeholders
    content = content.replace('<start>', 'STARTTOKENPLACEHOLDER')
    content = content.replace('<end>', 'ENDTOKENPLACEHOLDER')
    content = content.replace('<space>', 'SPACETOKENPLACEHOLDER')

    # Tokenize the content
    tokens = word_tokenize(content)

    # Replace placeholders back to original tokens
    tokens = ['<start>' if token == 'STARTTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<end>' if token == 'ENDTOKENPLACEHOLDER' else token for token in tokens]
    tokens = ['<space>' if token == 'SPACETOKENPLACEHOLDER' else token for token in tokens]

    # Map tokens to their corresponding values in the dictionary
    order = [dict_[token] for token in tokens if token in dict_]

    return order


def returns_batches(order, n):
    batches = [order[i:i + n] for i in range(len(order) - n + 1)]
    return batches


def rearrange(batches):
    X = []
    Y = []
    for i in range(len(batches) - 1):
        X.append(batches[i])
        Y.append(batches[i + 1])
    return X, Y


def split_list(lst, per):
    len_75 = int(len(lst) * per)
    first_list = lst[:len_75]
    second_list = lst[len_75:]
    return first_list, second_list




class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Apply mask to prevent attending to future tokens during decoding
        mask = tf.linalg.band_part(tf.ones_like(scaled_score), -1, 0)
        scaled_score -= 1e9 * (1 - mask)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights
    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None, training=None):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)

        return output
import keras
from keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=True)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=True)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
        })
        return config



class TokenAndPositionEmbedding(keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.token_emb = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)
        super().build(input_shape)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.embed_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        })
        return config

def build_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, dropout_rate, num_encoders, num_decoders):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(inputs)
    encoders_outputs = []
    x = embedding_layer
    for _ in range(int(num_encoders)):
        encoder_output = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
        encoders_outputs.append(encoder_output)

    decoder_inputs = layers.Input(shape=(maxlen,))
    y = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)(decoder_inputs)
    for _ in range(int(num_decoders)):
        for encoder_output in encoders_outputs:
            y = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(y)
    outputs = layers.Dense(vocab_size, activation="softmax")(y)

    model = keras.Model(inputs=[inputs, decoder_inputs], outputs=outputs)
    return model









import numpy as np

def transformer(maxlen, embed_dim, num_heads, ff_dim, num_blocks,content, dropout_rate, per, batch_size, epochs, num_encoders, num_decoders):
    # Read input file
    # with open(input_file, 'r', encoding='utf-8') as file:
    #     content = file.read()

    content=pad_segments(content=content,maxlen=maxlen)
    print('processed text for training')
    print(content)



    # Split and sort the content
    u = split_and_sort(string=content)

    # Create a dictionary of unique words
    dictionary = return_dict( unique_words=u)
    vocab_size=max(  list(dictionary.values() ) )+1
    # Return the order of the dictionary
    order = return_order(dict_=dictionary, content=content)

    batches = returns_batches(order=order, n=maxlen)

    # Rearrange batches into X and Y
    X, Y = rearrange(batches=batches)

    # Split data into train and test sets
    x_train, x_test = split_list(lst=X, per=per)
    y_train, y_test = split_list(lst=Y, per=per)
    x_train=np.array(X)
    x_test = np.array(x_test[10:20])
    y_train = np.array(Y)
    y_test = np.array(y_test[10:20])

    # Build the transformer model
    model = build_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, dropout_rate, num_encoders, num_decoders)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([x_train,x_train], y_train, validation_data=([x_test,x_test], y_test), batch_size=batch_size, epochs=epochs)

    # Make predictions
    predictions = model.predict([x_test, x_test])
    predicted_classes = np.argmax(predictions, axis=-1)
    predicted_words = number_to_words(predictions=predicted_classes, dictionary=dictionary)
    x_ = number_to_words(predictions=x_test, dictionary=dictionary)
    y_ = number_to_words(predictions=y_test, dictionary=dictionary)

    # Print results
    for i in range(len(predicted_words)):
        print('x_:')
        print(x_[i])
        print('y_')
        print(y_[i])
        print('predicted: ')
        print(predicted_words[i])

    return model, dictionary, maxlen



#model, dictionary, maxlen = transformer()


def query_gen_sentences(query, model, dictionary, maxlen):
    # Convert the query to the order of words based on the provided dictionary



    query_order = return_order(dict_=dictionary, content=query)
    u_order = np.array(query_order)

    # Pad the order to match the maximum length
    padding_length = max(0, maxlen - len(u_order))
    padded_u_order = np.pad(u_order, (0, padding_length), mode='constant', constant_values=0)
    padded_u_order = np.reshape(padded_u_order, (1, -1))

    # Generate predictions using the model
    # Assuming x_data_1 and x_data_2 are your input data tensors
    predictions = model.predict([padded_u_order, padded_u_order])
    predicted_classes = np.argmax(predictions, axis=-1)

    # Convert predicted classes to words using the provided dictionary
    words = number_to_words(predictions=predicted_classes, dictionary=dictionary)

    return words


#query_gen_sentences(query="the wellbeing of future generations this underscores the immense importance",
                    #model=model, dictionary=dictionary, maxlen=maxlen)

# transformer()



'''

    # Sample data
x_train = [
        [2, 5, 8, 10, 15, 7, 12, 3, 6, 20],
        [18, 1, 9, 14, 4, 11, 19, 13, 16, 17],
        [5, 3, 8, 12, 15, 2, 1, 7, 6, 10]
    ]

    # Assuming each element in y_train corresponds to the next number in x_train
y_train = [
        [5, 8, 10, 15, 7, 12, 3, 6, 20, 18],
        [1, 9, 14, 4, 11, 19, 13, 16, 17, 5],
        [3, 8, 12, 15, 2, 1, 7, 6, 10, 5]
    ]

x_test = [
        [15, 3, 8, 12, 9, 2, 1, 7, 15, 7]
    ]

y_test = [
        [3, 8, 12, 15, 2, 1, 7, 6, 10, 5]
    ]

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

maxlen = 10
vocab_size = 629
embed_dim = 256
num_heads = 8
ff_dim = 512
num_blocks = 4
dropout_rate = 0.1
num_encoders = 2
num_decoders = 2
model = build_transformer_model(maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, dropout_rate, num_encoders, num_decoders)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit([x_train, x_train], y_train, validation_data=([x_test, x_test], y_test), batch_size=64, epochs=20)
predictions = model.predict([x_test, x_test])
print("Predictions:", predictions)
predicted_classes = np.argmax(predictions, axis=-1)
print("Predicted Classes:", predicted_classes)


'''


