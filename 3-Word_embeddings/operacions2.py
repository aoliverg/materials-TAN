from gensim.models import Word2Vec
import gensim.downloader as api

model_gigaword = api.load("glove-wiki-gigaword-100")

similars=model_gigaword.most_similar(positive=['king','woman'],negative=['man'], topn=10)

for similar in similars:
    print(similar[0],similar[1])
