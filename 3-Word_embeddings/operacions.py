from gensim.models import Word2Vec
model = Word2Vec.load("CBOW.model")

similars=model.wv.most_similar(positive=['rei','dona'],negative=['home'], topn=10)

for similar in similars:
    print(similar[0],similar[1])
