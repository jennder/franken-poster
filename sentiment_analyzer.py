from nltk.sentiment import SentimentIntensityAnalyzer

"""
Performs sentiment analysis on a movie based on its dialogues
and can create a movie poster background based on the sentiment
of the conversations.
"""
class SentimentAnalyzer:

    def __init__(self, conversations):
        self.conversations = self.get_conversations(movie_id)

    def get_conversations(self, conversations):
        """Read the movie conversations file and get all lines associated with
        the given movie ID.

        String -> [Listof String]
        """
        pass
                

    
    def run_model(self):
        pass

"""
Using nltk's sentiment analyzer
"""
class VaderAnalyzer(SentimentAnalyzer):
    def run_model(self):
        pass

"""
Using TextBlob's Naive Bayes sentiment analyzer
"""
class NaiveAnalyzer(SentimentAnalyzer):
    def run_model(self):
        pass
