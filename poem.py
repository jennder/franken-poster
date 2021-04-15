import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers, optimizers
from keras.layers.experimental.preprocessing import TextVectorization
from keras.layers import Embedding, Dense, Dropout, Input, LSTM, GlobalMaxPool1D, Bidirectional
from keras.models import Sequential
from keras.initializers import Constant
from keras.utils import np_utils
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
import spacy

class PoemGen:

    def __init__(self, conversations, movie_id):
        self.conversations = conversations
        self.movie_id = movie_id
        self.sentences = []

    # def tokenize_words(self,input):
    #     # lowercase everything to standardize it
    #     input = input.lower()

    #     # instantiate the tokenizer
    #     tokenizer = RegexpTokenizer(r'\w+')
    #     tokens = tokenizer.tokenize(input)

    #     # if the created token isn't in the stop words, make it part of "filtered"
    #     filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
    #     return " ".join(filtered)

    def poem_generator(self, file, n_sents=4):
        nlp = spacy.load("en_core_web_md")
        # processed_inputs = self.tokenize_words("\n".join(self.conversations))

        # vectorizer = TextVectorization(max_tokens=30000, output_sequence_length=200)
        # text_ds = tf.data.Dataset.from_tensor_slices(self.conversations).batch(128)
        # vectorizer.adapt(text_ds)
        # vocab = vectorizer.get_vocabulary()

        #generate the embedding matrix
        # total_words = len(vocab)


        tokenizer = Tokenizer() #instantiating the tokenizer
        tokenizer.fit_on_texts(self.conversations) #creates tokens for each words 
        input_sequences = []
        for line in self.conversations:
            token_list = tokenizer.texts_to_sequences([line])[0] #converts each sentence as its tokenized equivalent
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1] #generating n gram sequences
            input_sequences.append(n_gram_sequence)
        max_sequence_len = max([len(x) for x in input_sequences])
        # print(tokenizer.word_index.keys())

        total_words = len(tokenizer.word_index) + 1

        embedding_dim = len(nlp('The').vector)
        embedding_matrix = np.zeros((total_words, embedding_dim))
        for i, word in enumerate(tokenizer.word_index.keys()):
            embedding_matrix[i] = nlp(word).vector
        print("Found %s word vectors." % len(embedding_matrix))
        print(token_list)
        print(input_sequences, len(input_sequences))

        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        #Load the embedding matrix as the weights matrix for the embedding layer and set trainable to False
        embedding_layer = Embedding(
            total_words,
            embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=False,
        )

        model = Sequential()
        model.add(embedding_layer)
        # model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        # model.add(Dropout(0.2))
        # model.add(LSTM(256, return_sequences=True))
        # model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(20)))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
        )
        xs, labels = input_sequences[:,:-1],input_sequences[:,-1] #creating xs and their labels using numpy slicing
        # flat_list = [item for sublist in labels for item in sublist]
        # print(labels, total_words, len(tokenizer.word_index) + 1)
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        model.fit(xs, ys, batch_size=128, epochs=2000)


        #predicting the next word using an initial sentence
        input_phrase = " ".join(self.conversations[0].split()[:5])
        next_words = 50
        
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([input_phrase])[0] #converting our input_phrase to tokens and excluding the out of vcabulary words
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') #padding the input_phrase
            predicted = model.predict_classes(token_list, verbose=0) #predicting the token of the next word using our trained model
            output_word = "" #initialising output word as blank at the beginning
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word #converting the token back to the corresponding word and storing it in the output_word
                    break
            input_phrase += " " + output_word
        print(input_phrase)
       