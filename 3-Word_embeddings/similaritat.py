from gensim.models import Word2Vec
model = Word2Vec.load("CBOW.model")
similaritat1=model.wv.similarity('gos', 'gat')
print("similaritat entre gos i gat:",similaritat1)
similaritat2=model.wv.similarity('gos', 'aixeta')
print("similaritat entre gos i aixeta:",similaritat2)