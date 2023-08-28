import os
import spacy
import pickle
from whatlies import EmbeddingSet
from whatlies.language import SpacyLanguage
from whatlies.transformers import Pca,Umap
import csv
import time

start = time.time()
nlp = spacy.load('en_core_web_lg')
print("load model ",(time.time()-start))
lang = SpacyLanguage("en_core_web_lg")
print("load model ",(time.time()-start))

savefile='words.pkl'

def gen_words():
    if os.path.exists(savefile):
        with open(savefile, 'rb') as pfile:
            words = pickle.load(pfile)
            return words
    words = set()
    count = 0
    with open('Shakespeare_data.csv') as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            try:
                tokens = (nlp(row[5].lower()))
                start = time.time()
                for token in tokens:
                    if token.text not in words:
                        words.add(token.text)
                        count += 1
                        if False and count > 10:
                            return words
            except Exception as e:
                pass
    with open(savefile,'wb') as pfile:
        pickle.dump(words,file)
    return words
        
start = time.time()
words = gen_words()
print("gen_words  ",(time.time()-start))
start = time.time()
emb = EmbeddingSet(*[lang[word] for word in words])
print("embeddings ",(time.time()-start))
start = time.time()
pca_plot = emb.transform(Pca(2)).plot_interactive()
print("plot pca   ",(time.time()-start))
start = time.time()
umap_plot = emb.transform(Umap(2)).plot_interactive()
print("plot umap  ",(time.time()-start))
start = time.time()

plots = pca_plot | umap_plot

plots.save('sp.html')
print("save plots ",(time.time()-start))
