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
    def __init__(self, nin, name=''):
        self.w = [Value(random.uniform(-1, 1)*0.01, name=f'{name}:w{i}') for i in range(nin)]
        self.b = Value(0, name=f'{name}:b')
        self.name = name

    def update_name(self, name):
        for i, value in enumerate(self.w):
            value.update_name(f'{name}:w{i}')
        self.b.update_name(f'{name}:b')

    def __call__(self, x):
        out = sum(w * x_ for w, x_ in zip(self.w, x)) + self.b
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Model(Module):
    def __init__(self, layers):
        for i, layer in enumerate(layers):
            layer.name = i
            layer.update_neuron_names()
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
    
    def __repr__(self):
        return f'Model({len(self.layers)} layers): ' + ' -> '.join([str(layer) for layer in self.layers])

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, name=''):
        self.embeddings = np.array([Value(random.uniform(-1, 1), name=f'Emb({name}):n{i}') for i in range(embedding_dim * num_embeddings)]).reshape((num_embeddings, embedding_dim))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name

    def update_neuron_names(self):
        for i, embedding in enumerate(self.embeddings[0]):
            embedding.update_name(f'L{self.name}:n{i}')

    def __call__(self, indices):
        return self.embeddings[np.array(indices, dtype=int)].flatten()

    def parameters(self):
        return self.embeddings.flatten()
    
    def __repr__(self) -> str:
        return f'Embedding(num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim})'
    
class Linear(Module):
    def __init__(self, nin, nout, activation='', name='', **kwargs):
        self.neurons = [Neuron(nin, name=f'L{name}:n{i}',**kwargs) for i in range(nout)] 
        self.activation = activation.upper()
        self.name = name

    def update_neuron_names(self):
        for i, neuron in enumerate(self.neurons):
            neuron.update_name(f'L{self.name}:n{i}')

    def __call__(self, x):
        out = [neuron(x).tanh() if self.activation == 'TANH' else neuron(x).relu() if self.activation == 'RELU' else neuron(x) for neuron in self.neurons]
        return out if len(out) > 1 else out[0]
    
    def parameters(self):
        return np.array([p for neuron in self.neurons for p in neuron.parameters()])
    
    def __repr__(self):
        return f'Linear({len(self.neurons)} neurons, activation={self.activation})'
    

# TODO: make a layer abstract class which Linear and Embedding extending it