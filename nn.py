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
        self.training = True
        self.shape = [layer.dims for layer in layers]
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                self.input_shape = layer.nin
                break
            elif isinstance(layer, Embedding):
                self.input_shape = int(self.layers[i+1].nin/layer.embedding_dim)
                break
        self.output_shape = self.shape[-1][1]

        self.layer_outs = None

    def __call__(self, ix):
        res = []
        self.layer_outs = [[Value(0) for _ in range(len(ix))] for _ in self.layers]
        for i, ixVal in enumerate(ix):
            x = [ixVal]
            for k, layer in enumerate(self.layers):
                if isinstance(layer, Linear) and layer.bn:
                    if i == 0:  # Only compute the batch normalization once for the entire batch
                        self.forward_batch(ix, k, layer)
                    x = self.layer_outs[k][i]  # Retrieve the normalized output for the current input
                else:
                    x = layer(x)  # Apply the layer to the current input
            res.append(x)

        return res[0] if len(res) == 1 else res

    def forward_batch(self, batch_inputs, layer_num, layer):
        batch_outputs = []
        for i, x in enumerate(batch_inputs):
            for layer in self.layers[: (layer_num+1)]:
                try:
                    x = layer([x] if not isinstance(x, list) and not isinstance(x, np.ndarray) else x)
                except Exception as e:
                    print(f"Error applying layer: {e}")
            batch_outputs.append(x)
        
        batch_outputs = np.array(batch_outputs) # Shape: (batch_size, nout)

        if self.training:
            # mean for all nodes in layer outputs, so, dim = 0
            means = sum(batch_outputs) * len(batch_outputs)**-1

            vars = sum([(x + means*-1)**2 for x in batch_outputs]) * len(batch_outputs)**-1

            normalized_outputs = (batch_outputs + means*-1) * (vars**(1/2) + 1e-10)**-1

            layer.moving_mean = layer.momentum * layer.moving_mean + (1-layer.momentum) * np.array([mean.data for mean in means])

            layer.moving_var = layer.momentum * layer.moving_var + (1-layer.momentum) * np.array([var.data for var in vars])

        else:
            normalized_outputs = (batch_outputs + layer.moving_mean*-1) * (layer.moving_var**(1/2) + 1e-10)**-1

        # Store the normalized outputs for this layer
        for i, output in enumerate(normalized_outputs):
            self.layer_outs[layer_num][i] = layer.gamma*output + layer.beta

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
        self.bn = bn
        if self.bn:
            self.gamma = [Value(1, name=f'{i}:{name}:gamma') for i in range(nout)]
            self.beta = [Value(0, name=f'{i}:{name}:beta') for i in range(nout)]
            self.moving_mean = np.array([0 for _ in range(nout)])
            self.moving_var = np.array([0 for _ in range(nout)])
            self.momentum = np.array([0.99 for _ in range(nout)])

    def update_neuron_names(self):
        for i, neuron in enumerate(self.neurons):
            neuron.update_name(f'L{self.name}:n{i}')

    def __call__(self, x):
        out = [neuron(x).tanh() if self.activation == 'Tanh' else neuron(x).relu() if self.activation == 'Relu' else neuron(x) for neuron in self.neurons]
        return out if len(out) > 1 else out[0]

    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        if self.bn:
            params += self.gamma + self.beta
        return np.array(params)

    def __repr__(self):
        return f'Linear(({self.nin}, {self.nout}) -> {self.activation})'