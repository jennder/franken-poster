import pandas as pd
import numpy as np
from keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential, load_model
from keras.initializers import Constant
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
import en_core_web_md
import re
import random

nlp = en_core_web_md.load()

"""
A class that trains a model to generate a poem based on the movie dialogue
of a given movie, and generates a poem to print on the movie poster.
"""
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
        tokenizer, max_sequence_len = self.model_setup()

        # Load existing model if it exists for the movie
        try:
            model = load_model('text/model_%s.h5' % self.movie_id)
            print("Existing model loaded")
        except:
            print("Training a new model. This may take a few minutes.")
            model = self.create_model(tokenizer)

        #predicting the next word using an initial sentence
        input_phrase = self.title
        for _ in range(self.POEM_LENGTH):
            token_list = tokenizer.texts_to_sequences([input_phrase])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            probs = model.predict(token_list)[0]
            predicted = np.random.choice(range(0, len(probs)), p=probs)
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
        """
        Sets up the model for generating a poem with a tokenizer, and splits the conversations
        into sequences.

        Void -> [Tupleof Tokenizer Int]
        """
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


    def create_model(self, tokenizer):
        """
        Create and train a new Keras RNN model using pre-trained word embeddings from SpaCy,
        and then training the model further on the movie conversation text. Saves the model
        to the text/ folder once training is completed
        
        Tokenizer -> Model
        """
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

        model.fit(xs, ys, batch_size=128, epochs=200)

        # Save model
        model.save('text/model_%s.h5' % self.movie_id)
        return model
