import pandas as pd
import numpy as np
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential, load_model
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pickle import dump
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
import random

class PoemGen:
    POEM_LENGTH = 50

    def __init__(self, conversations, movie_id, title):
        self.conversations = [conv.lower() for conv in conversations]
        self.movie_id = movie_id
        self.sentences = []
        self.title = title

    def poem_generator(self):
        """
        Generate a poem with POEM_LENGTH words following the movie title.

        Void -> String
        """
        # Load existing model if it exists for the movie
        try:
            model = load_model('text/model_%s.h5' % self.movie_id)
            print("loaded")
        except:
            model = self.create_model()

        #predicting the next word using an initial sentence
        input_phrase = self.title

        # TODO split this out
        for _ in range(self.POEM_LENGTH):
            tokenizer, max_sequence_len = self.model_setup()
            token_list = tokenizer.texts_to_sequences([input_phrase])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') 
            predicted = np.argmax(model.predict(token_list), axis=-1)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            input_phrase += " " + output_word
        poem = self.segment(input_phrase)
        return poem

    def segment(self, poem):
        """
        Add line breaks to the poem according to the rules of poetry.
        Rule based text processing
        # TODO what are the rules

        String -> String
        """
        
        line_length = random.randint(5, 10)
        index = 0
        generate = ""
        for word in poem.split():
            if index > line_length:
                generate += "\n"
                index = 0
                line_length = random.randint(5, 10)
            generate += " " + word
            index += self.count_syllables(word)

        return generate


    def count_syllables(self, word):
        """Count the syllables in the given word with regex.
        Look for a vowel followed by any number of consonants.

        String -> Natural
        """
        syllables = r'[aeiou][b-df-hj-np-tv-z]+'
        res = re.findall(syllables, word)
        count = len(res)
        return count if count > 0 else 1 # accounts for the, a, my

    
    def model_setup(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.conversations)
        tokenizer.fit_on_sequences(self.conversations)

        # n-gram sequences
        self.input_sequences = []
        for line in self.conversations:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                self.input_sequences.append(n_gram_sequence)
        max_sequence_len = max([len(x) for x in self.input_sequences])
        self.input_sequences = np.array(pad_sequences(self.input_sequences, maxlen=max_sequence_len, padding='pre'))
        return (tokenizer, max_sequence_len)


    def create_model(self):
        #nlp = spacy.load("en_core_web_md")
       
        tokenizer, _ = self.model_setup()
        total_words = len(tokenizer.word_index) + 1

        # Use word embeddings from spacy
        embedding_dim = len(nlp('The').vector)
        embedding_matrix = np.zeros((total_words, embedding_dim))
        for i, word in enumerate(tokenizer.word_index.keys()):
            embedding_matrix[i] = nlp(word).vector

        embedding_layer = Embedding(
            total_words,
            embedding_dim,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=True,
        )
        model = Sequential()
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(20)))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["acc"]
        )
        xs, labels = self.input_sequences[:,:-1],self.input_sequences[:,-1]
        ys = to_categorical(labels, num_classes=total_words)

        model.fit(xs, ys, batch_size=128, epochs=150)

        # Save model
        dump(tokenizer, open('tokenizer_%s' % self.movie_id, 'wb'))
        model.save('text/model_%s.h5' % self.movie_id)

        return model
