from sentiment_analyzer import VaderAnalyzer, NaiveAnalyzer
from poem import PoemGen
from parser import Parser

class FrankenPoster():

    def __init__(self, movie_id):
        self.movie_id = movie_id
        parser = Parser(self.movie_id)
        self.conversation = parser.parse()
        self.title = parser.get_movie_title()

    def gen_two_posters(self):
        """
        Generate and save two movie posters.
        One will use the Vader sentiment model for the background and the other
        will use Naive Bayes. Both posters will have the same generated poem
        from the RNN.

        Void -> Void
        """
        print("Generating a poem...")
        poem = PoemGen(self.conversation, self.movie_id, self.title)
        text = poem.poem_generator()

        print("Generating the VADER poster...")
        self.gen_poster(VaderAnalyzer(self.conversation, self.movie_id), "vader", text)
        print("Generating the Naive Bayes poster...")
        self.gen_poster(NaiveAnalyzer(self.conversation, self.movie_id), "nb", text)

    def gen_poster(self, sent_model, model_name, text):
        """
        Generate one single movie poster for the given model with the given poem
        displayed as an SVG.
        """
        text_lines = [("<tspan x=\"36\" dy=\"28\">%s</tspan>" % t) for t in text.split("\n")]
        text_lines = "\n".join(text_lines)
        svg = f"""
<svg width="600" height="800" xmlns="http://www.w3.org/2000/svg">
    {sent_model.run_model()}
    <text x="36" y="200" fill="white" font-size="24">
        {text_lines}
    </text>
</svg>"""
        out_file = open('posters/%s_%s.svg' % (self.movie_id, model_name), 'w')
        out_file.write(svg)
        out_file.close()

if __name__ == "__main__":
    id = input("Enter movie id (as an int in [0, 616]): ")
    poster = FrankenPoster('m%s' % id)
    poster.gen_two_posters()
