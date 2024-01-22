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

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)*0.01) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        out = sum(w * x_ for w, x_ in zip(self.w, x)) + self.b
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Model(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, ix):
        res = []
        for ixVal in ix:
            x = [ixVal]
            for layer in self.layers:
                x = layer(x)
            res.append(x)
        return res[0] if len(res) == 1 else res
    
    def parameters(self):
        return np.array ([p for layer in self.layers for p in layer.parameters()])
    
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
    
    def __repr__(self) -> str:
        return f'Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})'
    
class Linear(Module):
    def __init__(self, nin, nout, activation='', **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)] 
        self.activation = activation.upper()
    def __call__(self, x):
        out = [neuron(x).tanh() if self.activation == 'TANH' else neuron(x).relu() if self.activation == 'RELU' else neuron(x) for neuron in self.neurons]
        return out if len(out) > 1 else out[0]
    
    def parameters(self):
        return np.array([p for neuron in self.neurons for p in neuron.parameters()])