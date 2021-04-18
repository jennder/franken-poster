import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

"""
Performs sentiment analysis on a movie based on its dialogues
and can create a movie poster background based on the sentiment
of the conversations.
"""
class SentimentAnalyzer:

    def __init__(self, conversations, movie_id):
        self.conversations = conversations
        self.movie_id = movie_id
        self.scores = []
    
    def run_model(self):
        pass

    def create_img(self):
        """
        Generate svg string from sentiment scores where rgb values correspond to
        neg, pos and neutral values respectively. Not a complete svg, only
        the background fill which will be added to the poster.
        scores: [Listof Number[-1, 1]], # scores for each individual conversation

        return: part of an SVG string representing a rectangle with gradient fill
        """
        return f"""
    <defs>
        <linearGradient id="Gradient{self.movie_id}" x1="0" x2="0" y1="0" y2="1">
            {self.__gradient_offsets()}
        </linearGradient>
    </defs>
        <rect width="600" height="800" fill="url(#Gradient{self.movie_id})"/>
        """
    
    def __gradient_offsets(self):
        """
        Generates offset stop tags for the gradient rectangle background.
        """
        tags = []
        offset = 0
        d_offset = 100 / (len(self.scores) - 1)
        for idx, s in enumerate(self.scores):
            red, green, blue = self.__get_color(s)
            if (idx == len(self.scores) - 1):
                offset = 100
            tags.append(f"""<stop offset="{offset}%" stop-color="rgb({red}, {green}, {blue})"/>""")
            offset += d_offset
        return tags
    
    def __get_color(self, movie_sentiment):
        """Computer RGB color based on the compound sentiment score in the range [-1,1].
        The returned rgb color will fall on a linear gradient of red being the most 
        negative sentiment (-1), blue if neutral (0), and green for most positive sentiment(1).
        """
        red = max(0, movie_sentiment * 255 * -1)
        blue = 0
        if movie_sentiment < 0:
            blue = (1 + movie_sentiment) * 255
        else:
            blue = (1 - movie_sentiment) * 255
        green = max(0, 255 * movie_sentiment)
        return (red, green, blue)


"""
Using nltk's sentiment analyzer
"""
class VaderAnalyzer(SentimentAnalyzer):
    def __init__(self, conversations, movie_id):
        super().__init__(conversations, movie_id)
        nltk.download('vader_lexicon')

    def run_model(self):
        sia = SentimentIntensityAnalyzer()
        self.scores = [sia.polarity_scores(conv)["compound"] for conv in self.conversations]
        return self.create_img()
        

"""
Using TextBlob's Naive Bayes sentiment analyzer
"""
class NaiveAnalyzer(SentimentAnalyzer):
    def run_model(self):
        self.scores = []
        for conv in self.conversations:
            blob = TextBlob(conv)
            sent = blob.sentiment
            self.scores.append(sent.polarity)
        return self.create_img()
