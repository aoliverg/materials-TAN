# tanh function in Python
# from: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 50)
z = np.tanh(x)

plt.subplots(figsize=(8, 5))
plt.plot(x, z)
plt.grid()
plt.title("Tangent hiperbòlica")
plt.ylabel("Activació F(x)")
plt.xlabel("Estímul x")
plt.show()
