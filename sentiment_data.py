from sentiment_analyzer import VaderAnalyzer, NaiveAnalyzer
from parser import Parser
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

def get_data():
    """
    Save the sentiment scores of the first ten movies in the corpus to a csv file using both models.
    """
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

def graph_single():
    """
    Create a scatterplot for the sentiment scores of a single movie for both
    Vader and Naive Bayes models.
    """
    df = pd.read_csv('data/sentiment_m0.csv')
    data = df.to_dict(orient="list")
    v = plt.scatter(data['conv_id'], data['vader'], color="pink", alpha=0.5)
    n = plt.scatter(data['conv_id'], data['nb'], color="turquoise", alpha=0.5)
    plt.title("Sentiment scores of conversations in 10 movies")
    plt.xlabel("movie")
    plt.ylabel("sentiment score")
    plt.legend([v, n], ["Vader", "Naive Bayes"])
    plt.show()
    plt.savefig("m0_sentiment.png")

def bin(val):
    """
    Returns the categorical value of the given sentiment score.
    """
    if val < -0.6:
        return "very_neg"
    elif val < -0.2 and val >= -0.6:
        return "slightly_neg"
    elif val < 0.2:
        return "neutral"
    elif val < 0.6:
        return "slightly_pos"
    else:
        return "very_pos"

def graph_group():
    """
    Create a graph comparing the sentiment value results of vader vs naive bayes models
    by categorically sorting each value into one of five categories ranging from
    very negative to very positive. Groups this for the first 10 movies in the corpus.
    """
    order = ["very_neg", "slightly_neg", "neutral", "slightly_pos", "very_pos"]
    color_list = ["crimson", "coral", "cornflowerblue", "darkcyan", "chartreuse"]
    df = pd.read_csv('data/sentiment.csv')
    
    df["vader_bin"] = df["vader"].map(bin)
    df["nb_bin"] = df["nb"].map(bin)

    df_v_grp = df[['vader_bin', 'movie_id', "vader"]]
    df_v_grp.rename(columns={"vader_bin": "bin", "vader": "sentiment"}, inplace=True)
    df_v_grp = df_v_grp.groupby(["movie_id", 'bin']).count()
    df_v_grp.reset_index(inplace=True)

    df_nb_grp = df[['nb_bin', 'movie_id', "nb"]]
    df_nb_grp.rename(columns={"nb_bin": "bin", "nb": "sentiment"}, inplace=True)
    df_nb_grp = df_nb_grp.groupby(["movie_id", 'bin']).count()
    df_nb_grp.reset_index(inplace=True)
    df_nb_grp["movie_id"] = df_nb_grp["movie_id"].map(lambda x: x + 0.3)

    big_df = pd.concat([df_v_grp, df_nb_grp])
    fig = px.bar(big_df, x="movie_id", y="sentiment", color="bin",
                barmode = 'stack', category_orders={"bin": order},
                color_discrete_sequence=color_list, title="Distribution of conversation sentiment scores",
                labels={"movie_id": "movie", "sentiment": "conversation count"})
    cols = [0, 0.3, 1, 1.3, 2, 2.3, 3, 3.3, 4, 4.3, 5, 5.3, 6, 6.3, 7, 7.3, 8, 8.3, 9, 9.3]
    fig.update_layout(
        xaxis = dict(
            tickmode = 'array',
            tickvals = cols,
            ticktext = ["m%d %s" % (val // 1, "vader" if i % 2 == 0 else "nb")for i, val in enumerate(cols)]
        ),
        legend_title_text="Sentiment values"
    )
    fig.show()
