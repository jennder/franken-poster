import json
from sentiment_analyzer import VaderAnalyzer, NaiveAnalyzer
from poem import PoemGen

"""
A class to parse information from the movie dialogues corpus
"""
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


for i in range(0, 1):
    id = "m%d" % i
    print(id)
    parser = Parser(id)
    conversation = parser.parse()
    # vader = VaderAnalyzer(conversation, id)
    # vader.run_model()

    # nb = NaiveAnalyzer(conversation, id)
    # nb.run_model()

    poem = PoemGen(conversation, id)
    # poem.docs_to_sentences("%s.csv" % id)
    print(poem.poem_generator("text/sentences_%s.csv" % id))
