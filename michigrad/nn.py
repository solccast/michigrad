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
        if isinstance(x, list):
            return [xi.relu() for xi in x]
        return x.relu()
    

class Tanh(Module):
    """ 
        Implementación de la clase Tanh como capa de la red neuronal
    """
    def __call__(self, x):
        if isinstance(x, list):
            return [xi.tanh() for xi in x]
        return x.tanh()
    
class Sigmoid(Module):
    """ 
        Implementación de la clase Sigmoid como capa de la red neuronal
    """
    def __call__(self, x):
      if isinstance(x, list):
        return [xi.sigmoid() for xi in x]
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

class Linear(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    Modificación de la clase MLP para permitir capas de activación personalizadas: ReLU, Tanh, Sigmoid los cuales deben pasarse por parámetro al instanciar el modelo.
    """
    def __init__(self, nin, nouts, activations=[ReLU()]): # Hasta ahora el modelo se instancia así: xor = MLP(2, [3,3,1]), pero ahora se busca colocar las capas de activación (Relu, Tanh, Sigmoid) también, osea: xor =  MLP (2, [3, 3, 1], activations=[ReLU(), Tanh(), ReLU()])
        sz = [nin] + nouts
        self.activations = activations #Por defecto lo dejamos con ReLU
        self.act = 0
        self.layers = [Linear(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            if len(self.activations) == 1:
                 x = self.activations[0](x)
            else:
                x = self.activations[self.act](x)
                self.act += 1
        
        self.act = 0
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"