# Sigmoid function in Python
# from: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 50)
z = 1/(1 + np.exp(-x))

plt.subplots(figsize=(8, 5))
plt.plot(x, z)
plt.grid()
plt.title("Sigmoide")
plt.ylabel("Activació F(x)")
plt.xlabel("Estímul x")
plt.show()
