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

"""
Using nltk's sentiment analyzer
"""
class VaderAnalyzer(SentimentAnalyzer):
    def __init__(self, conversations, movie_id):
        super().__init__(conversations, movie_id)
        #nltk.download('vader_lexicon')

    def run_model(self):
        sia = SentimentIntensityAnalyzer()
        scores = [sia.polarity_scores(conv) for conv in self.conversations]
        self.__create_img(scores)
        
    def __create_img(self, scores):
        """
        Generate svg string from sentiment scores where rgb values correspond to
        neg, pos and neutral values respectively.
        """
        svg = f"""
<svg width="600" height="800" xmlns="http://www.w3.org/2000/svg">
    <defs>
        <linearGradient id="Gradient{self.movie_id}" x1="0" x2="0" y1="0" y2="1">
            {self.__gradient_offsets(scores)}
        </linearGradient>
    </defs>
        <rect width="600" height="800" fill="url(#Gradient{self.movie_id})"/>
</svg>"""
        out_file = open('img/%s.svg' % self.movie_id, 'w')
        out_file.write(svg)
        out_file.close()
    
    def __gradient_offsets(self, scores):
        tags = []
        offset = 0
        d_offset = 100 / (len(scores) - 1)
        for idx, s in enumerate(scores):
            if (idx == len(scores) - 1):
                offset = 100
            tags.append(f"""<stop offset="{offset}%" stop-color="rgb({s["neg"] * 256}, {s["pos"] * 256}, {s["neu"] * 256})"/>""")
            offset += d_offset
        return tags

"""
Using TextBlob's Naive Bayes sentiment analyzer
"""
class NaiveAnalyzer(SentimentAnalyzer):

    def run_model(self):
        scores = []
        for conv in self.conversations[0:5]:
            print("conv")
            blob = TextBlob(conv, analyzer=NaiveBayesAnalyzer())
            sent = blob.sentiment
            format_scores = {"neg": sent.p_neg, "pos": sent.p_pos}
            scores.append(format_scores)
        self.__create_img(scores)

    def __create_img(self, scores):
        print(scores)
        print(len(scores))

