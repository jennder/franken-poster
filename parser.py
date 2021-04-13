import json
from sentiment_analyzer import VaderAnalyzer, NaiveAnalyzer
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
class Parser:
    DELIMITER = " +++$+++ "

    def __init__(self, movie_id):
        self.movie_id = movie_id

    def parse(self):
        movie_found = False
        line_conv = {}
        conversations = []
        with open("movie-dialogs-corpus/movie_lines.txt", "r", encoding="utf-8", errors="ignore") as f:
            for entry in f:
                #L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!
                line = entry.split(self.DELIMITER)
                # print(line)
                line_id, _, movie, _, dialogue = line
                if movie == self.movie_id:
                    movie_found = True
                    line_conv[line_id] = dialogue.strip()
                elif movie_found:
                    break

        movie_found = False
        with open("movie-dialogs-corpus/movie_conversations.txt", "r", encoding="utf-8", errors="ignore") as f:
            for entry in f:
                #u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L198', 'L199']
                line = entry.split(self.DELIMITER)
                _, _, movie, line_ids = line
                if movie == self.movie_id:
                    movie_found = True
                    ids = json.loads(line_ids)
                    conversations.append(" ".join([line_conv[id] for id in ids]))
                elif movie_found:
                    break

        return conversations

parser = Parser("m0")
conversation = parser.parse()
vader = VaderAnalyzer(conversation, "m0")
vader.run_model()

nb = NaiveAnalyzer(conversation)
nb.run_model()

