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

def generate_text(s1):
    words = query_gen_sentences(query=s1,
                                model=main_model, dictionary=dictionary_1, maxlen=maxlen_1)

    for i in range(2):
        w1 = query_gen_sentences(query=words[-1],
                                 model=main_model, dictionary=dictionary_1, maxlen=maxlen_1)
        words.append(w1[0])
    generated_text = ' '.join(words)

    return generated_text




while True:
    i=input("Enter : ")
    o=generate_text(s1=i)
    print(o)


