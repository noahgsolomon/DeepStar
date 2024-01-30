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

    def __repr__(self):
        return f'Neuron({len(self.w)})'

class Model(Module):
    def __init__(self, layers):
        for i, layer in enumerate(layers):
            layer.name = i
            layer.update_neuron_names()
        self.layers = layers
        self.shape = [layer.dims for layer in layers]
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                self.input_shape = layer.nin
                break
            elif isinstance(layer, Embedding):
                self.input_shape = int(self.layers[i+1].nin/layer.embedding_dim)
                break
        self.output_shape = self.shape[-1][1]

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
        self.dims = (num_embeddings, embedding_dim)

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
    def __init__(self, nin, nout, activation='', name='', bn=False, **kwargs):
        self.neurons = [Neuron(nin, name=f'L{name}:n{i}',**kwargs) for i in range(nout)]
        self.activation = activation
        self.name = name
        self.nin = nin
        self.nout = nout
        self.dims = (nin, nout)

    def update_neuron_names(self):
        for i, neuron in enumerate(self.neurons):
            neuron.update_name(f'L{self.name}:n{i}')

    def __call__(self, x):
        out = [neuron(x).tanh() if self.activation == 'Tanh' else neuron(x).relu() if self.activation == 'Relu' else neuron(x) for neuron in self.neurons]
        return out if len(out) > 1 else out[0]

    def parameters(self):
        return np.array([p for neuron in self.neurons for p in neuron.parameters()])

    def __repr__(self):
        return f'Linear(({self.nin}, {self.nout}) -> {self.activation})'

class BatchNorm(Module):
    def __init__(self, nin, momentum=0.1, name=''):
        super().__init__()
        self.gamma = Value(1, name=f'BN{name}:gamma')
        self.beta = Value(0, name=f'BN{name}:beta')
        self.running_mean = np.zeros(nin)
        self.running_var = np.ones(nin)
        self.momentum = momentum
        self.training = True
        self.nin = nin
        self.name = name
        self.dims = (nin, 1)

    def __call__(self, x):
        if self.training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            self.running_mean = self.momentum * batch_mean + (1 - self.momentum) * self.running_mean
            self.running_var = self.momentum * batch_var + (1 - self.momentum) * self.running_var
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + 1e-10)
        else:
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + 1e-10)
        
        out = self.gamma * x_normalized + self.beta
        return out

    def parameters(self):
        return np.array([self.gamma, self.beta])

    def set_training(self, training):
        self.training = training

    def __repr__(self):
        return f'BatchNorm({self.nin})'



# TODO: make a layer abstract class which Linear and Embedding extending it
