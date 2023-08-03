import codecs

dictionary=[]

entrada=codecs.open("cat.dict","r",encoding="utf-8")

for linia in entrada:
    linia=linia.rstrip()
    dictionary.append(linia)

one_hot=[0] * len(dictionary)    
    
paraula="zumzejàvem"

posicio=dictionary.index(paraula)

one_hot[posicio-1]=1

print(one_hot)


