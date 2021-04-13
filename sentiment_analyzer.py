import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

"""
Performs sentiment analysis on a movie based on its dialogues
and can create a movie poster background based on the sentiment
of the conversations.
"""
class SentimentAnalyzer:

    def __init__(self, conversations):
        self.conversations = conversations
    
    def run_model(self):
        pass

"""
Using nltk's sentiment analyzer
"""
class VaderAnalyzer(SentimentAnalyzer):
    def __init__(self, conversations):
        super().__init__(conversations)
        nltk.download('vader_lexicon')

    def run_model(self):
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(conv) for conv in self.conversations]
        self.__create_img(scores)
        
    def __create_img(self, scores):
        """
        Generate svg string from sentiment scores where rgb values correspond to
        neg, pos and neutral values respectively.
        """
        pass
        # svg = """
        #     <svg width="300" height="400" xmlns="http://www.w3.org/2000/svg">
        #     <circle cx="25" cy="25" r="20"/>
        #     </svg>"""

"""
Using TextBlob's Naive Bayes sentiment analyzer
"""
class NaiveAnalyzer(SentimentAnalyzer):
    def run_model(self):
        pass
