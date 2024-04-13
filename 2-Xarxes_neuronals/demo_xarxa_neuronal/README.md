# Demo de xarxa neuronal

Aquesta demo està adaptada de [PyTorch Tutorial: Building a Simple Neural Network From Scratch](https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch). L'únic que s'ha fet és posar tot el codi en un únic arxiu que fa l'entrenament, la inferència i l'avaluació de la xarxa. També presentem un molt breu resum del contingut. Els lectors interessats en comprendre a fons el programa hauran d'accedir directament a la font original. Quan executis el programa, fixa't en les explicacions que apareixen per pantalla i observa les figures que apareixeran.

Aquest programa crea un xarxa neuronal senzilla fent servir la llibreria Pytorch. El conjunt de dades es crea artificialment amb la funció make_circles de la llibreria sklearn i consisteix en un conjunt de cerrcles que tenen associats o un valor 1 o un valor 0 i que estan disposats en un espai de dos dimensions formant un cercle més gran. La xarxa neuronal s'entrena per predir el valor d'un cercle a partir de les seves coordenades, és a dir, en funció de dos valors (característiques o *features*), la coordenada x i la y.

Quan executis el programa, observa bé el gràfic que mostra el conjunt de dades d'entrenament i de test (aquí podeu suposar que el color fosc és un valor d'1 i el clar de 0) i anoteu la precisió que assoleix la xarxa neuronal. Després, editeu el programa per canviar el valor del control sobre el soroll canviant en la línia 42 el noise=0.05 per noise=0.1. Com canvien les dades d'entrenament i test? I la precisió de la xarxa?


