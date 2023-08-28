import spacy
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca
import csv

nlp = spacy.load('en_core_web_lg')
lang = SpacyLanguage("en_core_web_lg")


def gen_words():
    words = set()
    with open('Shakespeare_data.csv') as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                tokens = (nlp(row[5]))
                for token in tokens:
                    if token.text not in words:
                        words.add(token.text)
            except e:
                pass
    return words
        
words = gen_words()
emb = EmbeddingSet(*[lang[word] for word in words])
pca_plot = emb.transform(Pca(2)).plot_interactive()
pca_plot.save('sp.html')
