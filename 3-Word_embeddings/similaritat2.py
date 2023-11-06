from gensim.models import Word2Vec
import gensim.downloader as api

model_gigaword = api.load("glove-wiki-gigaword-100")
similaritat1=model_gigaword.similarity('dog', 'cat')
print("similaritat entre dog i cat:",similaritat1)
similaritat2=model_gigaword.similarity('dog', 'tap')
print("similaritat entre dog i tap:",similaritat2)