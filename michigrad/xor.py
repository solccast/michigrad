from .nn import MLP
from .visualize import show_graph
import os

os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Red neuronal simple para resolver XOR (2 capas)
Model = MLP(2, [2, 1])

# Datos de entrada para la compuerta XOR
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [0, 1, 1, 0]

#Forward
yhat = [Model(x) for x in xs]
#Loss
loss = sum([(y - yh)**2 for y, yh in zip(ys, yhat)])/4
print(loss)
print("################################################")
# Setear grad a 0
for p in Model.parameters():
    p.grad = 0.0
#Backward
loss.backward()

for p in Model.parameters():
    p.data -= p.grad * 0.05

for _ in range(10000):
    #Forward
    yhat = [Model(x) for x in xs]
    #Loss
    loss = sum([(y - yh)**2 for y, yh in zip(ys, yhat)])/4
    #Zero
    for p in Model.parameters():
        p.grad = 0.0
    #Backward
    loss.backward()
    #Update
    for p in Model.parameters():
        p.data -= p.grad * 0.05
show_graph(loss)

print("######################PREDICTS##########################")
print(Model([0, 0]))
print(Model([0, 1]))
print(Model([1, 0]))
print(Model([1, 1]))