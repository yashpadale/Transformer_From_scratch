from transformer import transformer,query_gen_sentences


main_model, dictionary_1, maxlen_1 = transformer(maxlen=100,
                                                 embed_dim=256,
                                                 num_heads=16,
                                                 ff_dim=64,
                                                 num_blocks=6,
                                                 dropout_rate=0.1,
                                                 input_file='train.txt',
                                                 per=0.85,
                                                 batch_size=64,
                                                 epochs=15,
                                                 num_decoders=1,num_encoders=1,
                                                 window=10)

def generate_text(s1,main_model,dictionary_1,maxlen_1):
    i = '<start> ' + s1 + ' <end>'
    s1 = pad_segments(i, maxlen_1)
    words = query_gen_sentences(query=s1,
                                model=main_model, dictionary=dictionary_1, maxlen=maxlen_1)
    s=s1.split(' ')
    for i in range(len(s)):
        w1 = query_gen_sentences(query=words[-1],
                                 model=main_model, dictionary=dictionary_1, maxlen=maxlen_1)
        words.append(w1[0])
    #generated_text = ' '.join(words)

    return words[-1]

model.save('transformer_model.h5')
with open('dictionary.pkl', 'wb') as f:
    pickle.dump(dictionary, f)





custom_objects = {
    'MultiHeadSelfAttention': MultiHeadSelfAttention,
    'TransformerBlock': TransformerBlock,
    'TokenAndPositionEmbedding': TokenAndPositionEmbedding
}
loaded_model = load_model('transformer_model.h5', custom_objects=custom_objects)
with open('dictionary.pkl', 'rb') as f:
    loaded_dictionary = pickle.load(f)


while True:
    i=input("Enter : ")
    o=generate_text(s1=i)
    print(o)


