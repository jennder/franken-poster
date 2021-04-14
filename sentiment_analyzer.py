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
    
    def run_model(self):
        pass

    def create_img(self, movie_sentiment, scores, folder):
        """
        Generate svg string from sentiment scores where rgb values correspond to
        neg, pos and neutral values respectively.
        scores: [Listof Number[-1, 1]], # scores for each individual conversation
        movie_sentiment: Number[-1, 1] # composite score for the entire movie
        """
        svg = f"""
<svg width="600" height="800" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="Gradient{self.movie_id}" x1="0" x2="0" y1="0" y2="1">
            {self.__gradient_offsets(movie_sentiment, scores)}
        </linearGradient>
    </defs>
        <rect width="600" height="800" fill="url(#Gradient{self.movie_id})"/>
</svg>"""
        out_file = open('img/%s/%s.svg' % (folder, self.movie_id), 'w')
        out_file.write(svg)
        out_file.close()
    
    def __gradient_offsets(self, movie_sentiment, scores):
        red, green, blue = self.__get_color(movie_sentiment)
        tags = []
        offset = 0
        d_offset = 100 / (len(scores) - 1)
        for idx, s in enumerate(scores):
            if (idx == len(scores) - 1):
                offset = 100
            tags.append(f"""<stop offset="{offset}%" stop-color="rgb({red}, {green}, {blue})" stop-opacity="{(s + 1) / 2}"/>""")
            offset += d_offset
        return tags
    
    def __get_color(self, movie_sentiment):
        """Computer RGB color based on the movie sentiment score.
        """
        red = max(0, movie_sentiment * 255 * -1)
        blue = 0
        if movie_sentiment < 0:
            blue = (1 + movie_sentiment) * 255
        else:
            blue = (1 - movie_sentiment) * 255
        green = max(0, 255 * movie_sentiment)
        print("sentiment", movie_sentiment, "color", red, blue, green)
        return (red, green, blue)



"""
Using nltk's sentiment analyzer
"""
class VaderAnalyzer(SentimentAnalyzer):
    def __init__(self, conversations, movie_id):
        super().__init__(conversations, movie_id)
        #nltk.download('vader_lexicon')

    def run_model(self):
        entire_movie = " ".join(self.conversations)
        sia = SentimentIntensityAnalyzer()
        movie_sentiment = sia.polarity_scores(entire_movie)["compound"]
        scores = [sia.polarity_scores(conv)["compound"] for conv in self.conversations]
        self.create_img(movie_sentiment, scores, "vader")
        

"""
Using TextBlob's Naive Bayes sentiment analyzer
"""
class NaiveAnalyzer(SentimentAnalyzer):

    def run_model(self):
        scores = []
        entire_movie = " ".join(self.conversations)
        entire_blob = TextBlob(entire_movie)
        movie_sentiment = entire_blob.sentiment.polarity
        for conv in self.conversations:
            blob = TextBlob(conv)
            sent = blob.sentiment
            scores.append(sent.polarity)
        self.create_img(movie_sentiment, scores, "nb")

