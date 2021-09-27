import imageio
import random
from pathlib import Path
from wordcloud import WordCloud


class MyWordCloud:

    def __init__(self, corpus, time, print=True, font="SourceHanSansHC-Medium.otf"):
        self.corpus = corpus
        self.time = time
        self.print = print
        self.font = font

    def plot(self):

        font_path = Path("fonts", "SourceHanSansHC", self.font)
        back_img = imageio.imread(Path("images", "hkmap.png"))
        seed = 1234

        def grey_color_func(*args, **kwargs):
            random.seed(seed)
            return "hsl(0, 0%%, %d%%)" % random.randint(40, 60)

        wc = WordCloud(font_path=str(font_path), background_color='#FCFAF2',
                       max_words=1000, mask=back_img, max_font_size=200,
                       random_state=seed, width=1500, height=1200, margin=2)
        wc.generate(self.corpus)
        wc.recolor(color_func=grey_color_func)
        if self.print:
            wc.to_file(Path("outputs", "figs", f"cue_wordcloud_{self.time}.png"))
