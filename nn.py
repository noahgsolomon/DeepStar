import numpy as np
from value import Value
import random

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return np.array([])

class Neuron(Module):

    def __init__(self, nin, nonlinear=True):
        self.w = [Value(random.uniform(-1, 1)*0.01) for _ in range(nin)]
        self.b = Value(0)
        self.nonlinear = nonlinear

    def __call__(self, x):
        out = sum(w * x_ for w, x_ in zip(self.w, x)) + self.b
        return out.tanh() if self.nonlinear else out
    
    def parameters(self):
        return self.w + [self.b]

class Model(Module):
    def __init__(self, layers, batch_size=1):
        self.batch_size = batch_size
        self.layers = layers

    def __call__(self, xOG):
        if len(xOG) != self.batch_size:
            raise Exception(f'input of size: {len(xOG)} does not match batch size of {self.batch_size}')
        res = []
        for i in range(self.batch_size):
            x = [xOG[i]]
            for j, layer in enumerate(self.layers):
                x = layer(x)
            res.append(x)
        return res[0] if len(res) == 1 else res
    
    def parameters(self):
        return np.array([p for layer in self.layers for p in layer.parameters()])
    
    def parameters_val(self):
        return np.array([p.data for layer in self.layers for p in layer.parameters()])

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        self.embeddings = np.array([Value(random.uniform(-1, 1)) for _ in range(embedding_dim * num_embeddings)]).reshape((num_embeddings, embedding_dim))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def __call__(self, indices):
        return self.embeddings[np.array(indices, dtype=int)].flatten()

    def parameters(self):
        return self.embeddings.flatten()
    
class Linear(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)] 

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]  # This remains a list comprehension
        return out if len(out) > 1 else out[0]
    
    def parameters(self):
        return np.array([p for neuron in self.neurons for p in neuron.parameters()])
