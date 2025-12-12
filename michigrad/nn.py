import random
from michigrad.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class ReLU(Module):
    """ 
        Implementación de la clase ReLU como capa de la red neuronal
    """
    def __call__(self, x):
        return x.relu()
    

class Tanh(Module):
    """ 
        Implementación de la clase Tanh como capa de la red neuronal
    """
    def __call__(self, x):
        return x.tahn()
    
class Sigmoid(Module):
    """ 
        Implementación de la clase Sigmoid como capa de la red neuronal
    """
    def __call__(self, x):
        return (x.sigmoid())
    

class Neuron(Module):

    def __init__(self, nin): 
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = False # Por defecto queda lineal

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts): # Hasta ahoara el modelo se instancia así: xor = MLP(2, [3,3,1]), pero ahora se busca colocar las capas de activación (Relu, Tanh, Sigmoid) también, osea: xor =  MLP (2, [3, ])
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"