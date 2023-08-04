import gensim
from gensim.models import Word2Vec
import codecs

entrada=codecs.open("corpus-tok-cat.txt","r",encoding="utf-8")

data=[]

for linia in entrada:
    linia=linia.rstrip()
    tokens=linia.split(" ")
    data.append(tokens)
    
modelSkipGram = gensim.models.Word2Vec(data, min_count = 1, vector_size = 512, window = 5, sg = 1)
modelSkipGram.save("SkipGram.model")
modelSkipGram.wv.save_word2vec_format("SkipGram.wordvectors", binary= False)