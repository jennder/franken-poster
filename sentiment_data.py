from sentiment_analyzer import VaderAnalyzer, NaiveAnalyzer
from parser import Parser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_data():
    data = { 'movie_id': [], "conv_id": [], "vader": [], "nb": [] }
    for i in range(0, 10):
        parser = Parser("m%d" % i)
        conversation = parser.parse()

        vader = VaderAnalyzer(conversation, i)
        nb = NaiveAnalyzer(conversation, i)
        vader.run_model()
        nb.run_model()

        num_records = len(conversation)
        data["movie_id"] = data["movie_id"] + [i] * num_records
        data["conv_id"] = data["conv_id"] + list(range(0, num_records))
        data["vader"] = data["vader"] + vader.scores
        data["nb"] = data["nb"] + nb.scores

    df = pd.DataFrame(data=data)
    df.to_csv("data/sentiment.csv")
    return df

def graph():
    df = pd.read_csv('data/sentiment_m0.csv')
    data = df.to_dict(orient="list")
    plt.scatter(data['conv_id'], data['vader'], color="pink")
    plt.scatter(data['conv_id'], data['nb'], color="lightblue")
    plt.show()

graph()