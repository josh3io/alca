import os
import argparse
import spacy
import pickle
from whatlies.embeddingset import EmbeddingSet
from whatlies.embedding import Embedding
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca,Umap
from nltk.stem.snowball import SnowballStemmer
import csv


class Alca():
    def __init__(self,args):
        self.args = args
        self.model = args.model
        self.play = args.play
        self.sourcefile = args.sourcefile
        self.stem = args.stem
        self.x = args.x
        self.y = args.y
        self.words = set()
        self.embeddings = []
        self.graphs = None

        self.savefile = self.file_prefix() + args.savefile
        if self.play:
            self.savefile = f"{self.play}_{self.savefile}"

        self.outfile = self.file_prefix([self.play,args.x,args.y]) + args.outfile

        self.nlp = self.load_model()
        self.stemmer = SnowballStemmer("english")

    def file_prefix(self,parts=[]):
        parts += [
            self.args.model,
            self.args.stem,
            self.args.play,
        ]
        return '_'.join([str(x) for x in parts]) + '_'

    def load_model(self):
        return spacy.load(self.model)

    def wordstem(self,word):
        return self.stemmer.stem(word)

    def tokenize_line(self,line):
        tokens = (self.nlp(line.lower()))
        for token in tokens:
            word = token.text
            if '--' not in word:
                if self.stem:
                    word = self.wordstem(word)
                if word not in self.words and len(word) > 2:
                    self.words.add(word)

    def gen_words(self):
        if self.savefile and os.path.exists(self.savefile):
            with open(self.savefile, 'rb') as pfile:
                self.words = pickle.load(pfile)
                print("Loaded saved words")
                return

        if not os.path.exists(self.sourcefile):
            raise Exception("can't find sourcefile")

        print("Generating words")
        with open(self.sourcefile) as csvfile:
            next(csvfile)
            reader = csv.reader(csvfile)
            for row in reader:
                if not self.play or (self.play and row[1] == self.play):
                    try:
                        self.tokenize_line(row[5])
                    except Exception as e:
                        raise e
        if len(self.words):
            with open(self.savefile,'wb') as pfile:
                pickle.dump(self.words,pfile)
        else:
            raise Exception("no words found")

    def make_embeddings(self):
        embfile = 'emb_'+self.savefile
        if os.path.exists(embfile):
            try:
                with open(embfile,'rb') as pfile:
                    self.emb = pickle.load(pfile)
                    return
            except Exception as e:
                pass
        print("Generating embeddings")
        #self.emb = EmbeddingSet(*[self.nlp[word] for word in self.words])
        self.emb = EmbeddingSet({t.text: Embedding(t.text, t.vector) for t in self.nlp.pipe(self.words)})
        with open(embfile,'wb') as pfile:
            pickle.dump(self.emb,pfile)


    def add_graph(self,graph):
        if not self.graphs:
            self.graphs = graph
        else:
            self.graphs |= graph

    def make_pca(self):
        print("making pca")
        self.add_graph(self.emb.transform(Pca(2)).plot_interactive())

    def make_umap(self):
        print("making umap")
        self.add_graph(self.emb.transform(Umap(2)).plot_interactive())

    def plot_axes(self):
        print(f"plotting axes {self.x},{self.y}")
        self.add_graph(self.emb.plot_interactive(x_axis=self.x,y_axis=self.y))

    def plot_matrix(self):
        print("plotting matrix")
        self.add_graph(self.emb.plot_interactive_matrix(0,1,2))

    def plot(self):
        self.graphs.save('output/'+self.outfile)
        print(f"Saved {self.outfile}")

if __name__  == '__main__':
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--model', type=str,default='en_core_web_lg',help='model')
    parser.add_argument('--savefile', type=str,default='words.pkl',help='')
    parser.add_argument('--sourcefile', type=str,default='Shakespeare_data.csv',help='')
    parser.add_argument('--outfile', type=str,default='sp.html',help='')
    parser.add_argument('--stem', action='store_true',help='')
    parser.add_argument('--force', action='store_true',help='')
    parser.add_argument('--x', type=str)
    parser.add_argument('--y', type=str)
    parser.add_argument('--matrix',action='store_true')
    parser.add_argument('--play', type=str)
    args = parser.parse_args()

    alca = Alca(args)
    alca.gen_words()
    alca.make_embeddings()
    if args.x and args.y:
        alca.plot_axes()
    elif args.matrix:
        alca.plot_matrix()
    else:
        alca.make_umap()
    alca.plot()
