import spacy
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca,Umap
import csv

nlp = spacy.load('en_core_web_lg')
lang = SpacyLanguage("en_core_web_lg")


def gen_words():
    words = set()
    count = 0
    with open('Shakespeare_data.csv') as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                tokens = (nlp(row[5].lower()))
                for token in tokens:
                    if token.text not in words:
                        words.add(token.text)
                        count += 1
                        if True and count > 100:
                            return word
            except Exception as e:
                pass
    return words
        
words = gen_words()
emb = EmbeddingSet(*[lang[word] for word in words])
pca_plot = emb.transform(Pca(2)).plot_interactive()
umap_plot = emb.transform(Umap(2)).plot_interactive()

plots = pca_plot | umap_plot

plots.save('sp.html')
