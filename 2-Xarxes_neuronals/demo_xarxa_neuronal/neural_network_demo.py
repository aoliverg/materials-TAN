#from: https://www.datacamp.com/tutorial/pytorch-tutorial-building-a-simple-neural-network-from-scratch

import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import itertools
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns

warnings.filterwarnings("ignore")

#!pip install torch -q

# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len
   
batch_size = 64

# Create a dataset with 10,000 samples.

print("Creació del conjunt de dades. Un total de 10.000 mostres amb un soroll del 0.05.")

X, y = make_circles(n_samples = 10000,
                    noise= 0.05,
                    random_state=26)


print("Dividim les 10.000 mostres en un conjunt de test de 3.300 mostres i la resta per a l'entrenament.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=26)

print("Mida entrenament:",len(X_train))
print("Mida test:       ",len(X_test))

print("Visualitzem les dades. El color del cercle és el valor que volem predir a partir de les característiques (features) 0 i 1. Posem que el color fosc és un valor 1 i el clar un valor 0.")

# Visualize the data.
fig, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
train_ax.set_title("Training Data")
train_ax.set_xlabel("Feature #0")
train_ax.set_ylabel("Feature #1")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("Feature #0")
test_ax.set_title("Testing data")
plt.show()

# Instantiate training and test data
train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# Check it's working
for batch, (X, y) in enumerate(train_dataloader):
    print(f"Batch: {batch+1}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    break


print("Construïm una xarxa neuronal amb 2 dimensions d'entrada, 10 ocultes i una de sortida.")

input_dim = 2
hidden_dim = 10
output_dim = 1

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.sigmoid(self.layer_2(x))

        return x
       
model = NeuralNetwork(input_dim, hidden_dim, output_dim)

learning_rate = 0.1

loss_fn = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100
loss_values = []

print("Comença l'entrenament de la xarxa neuronal.")

for epoch in range(num_epochs):
    for X, y in train_dataloader:
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        pred = model(X)
        loss = loss_fn(pred, y.unsqueeze(-1))
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()

print("Acaba l'entrenament de la xarxa neuronal.")


print("Visualitzem el valor de la funció de pèrdua amb el temps.")
step = np.linspace(0, 100, 10500)

fig, ax = plt.subplots(figsize=(8,5))
plt.plot(step, np.array(loss_values))
plt.title("Step-wise Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

"""
Training Complete
"""

"""
We're not training so we don't need to calculate the gradients for our outputs
"""
print("Avaluem la xarxa neuronal fent servir el conjunt de dades de test.")
y_pred=[]
y_test=[]
total=0
correct=0
with torch.no_grad():
    for X, y in test_dataloader:
        outputs = model(X)
        predicted = np.where(outputs < 0.5, 0, 1)
        predicted = list(itertools.chain(*predicted))
        y_pred.append(predicted)
        y_test.append(y)
        total += y.size(0)
        correct += (predicted == y.numpy()).sum().item()

print(f'Precisió de la xarxa neuronal sobre els 3300 valors del conjunt de test: {100 * correct // total}%')

"""
Accuracy of the network on the 3300 test instances: 97%
"""

y_pred = list(itertools.chain(*y_pred))
y_test = list(itertools.chain(*y_test))

print(classification_report(y_test, y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)

plt.subplots(figsize=(8, 5))

sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")

plt.show()