import gensim
from gensim.models import Word2Vec
import codecs

entrada=codecs.open("corpus-tok-cat.txt","r",encoding="utf-8")

data=[]

for linia in entrada:
    linia=linia.rstrip()
    tokens=linia.split(" ")
    data.append(tokens)
    
modelCBOW = gensim.models.Word2Vec(data, min_count = 1,vector_size = 512, window = 5)
modelCBOW.save("CBOW.model")
modelCBOW.wv.save_word2vec_format("CBOW.wordvectors", binary= False)