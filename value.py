import math
import numpy as np

class Value:
    def __init__(self, data=1, children=[], _backward=lambda: None, _op='', name=''):
        self.data = np.float32(data)
        self.children = children
        self._backward = _backward
        self.grad = 0
        self._op = _op
        self.name = name
    
    def log(self):
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out = Value(math.log(self.data), children=[self], _backward=_backward, _op='log')
        return out

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), _op='+')

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        out = Value(0*self.data if self.data < 0 else self.data, children=[self], _op='Relu')

        def _backward():
            self.grad += (1 if out.data > 0 else 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        def _backward():
            self.grad += (1 - pow(out.data, 2)) * out.grad
        out = Value(math.tanh(self.data), children=[self], _backward=_backward, _op='Tanh')
        return out
    
    def sigmoid(self):
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out = Value(1/(1+math.pow(math.e, -self.data)), children=[self], _backward=_backward)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        if isinstance(other, (float, int)):
            def _backward():
                self.grad += other * (self.data ** (other - 1)) * out.grad
            out = Value(self.data ** other, children=[self], _backward=_backward, _op=f'**{other}')
        
        elif isinstance(other, Value):
            def _backward():
                self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
                other.grad += math.log(self.data) * (self.data ** other.data) * out.grad
            out = Value(self.data ** other.data, children=[self, other], _backward=_backward, _op=f'**{other.data}')
        
        return out
    
    def exp(self):
        def _backward():
            self.grad += math.exp(self.data) * out.grad
        out = Value(math.exp(self.data), children=[self], _backward=_backward, _op='exp')
        return out
    
    def __repr__(self):
        return f'name: {self.name} data: {self.data}, grad: {self.grad}, op: {self._op}'
    
    def backward(self):
        topo = []
        visited = set()

        def visit(node):
            if node not in visited:
                visited.add(node)
                for children in node.children:
                    visit(children)
                topo.append(node)

        self.grad = 1
        visit(self)
        for item in reversed(topo):
            item._backward()

    def __sub__(self, other):
            return self + (-other)
    
    def __rpow__(self, other):
        if isinstance(other, (float, int)):
            def _backward():
                self.grad += other ** self.data * math.log(other) * out.grad
            out = Value(other ** self.data, children=[self], _backward=_backward, _op=f'**{other}')
            return out

    def __rsub__(self, other):
        return (self*-1) + other
    
    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        if isinstance(other, Value):
            def _backward():
                self.grad += 1 / other.data * out.grad
                other.grad -= self.data / (other.data ** 2) * out.grad
            out = Value(self.data / other.data, children=[self, other], _backward=_backward, _op='÷')
            return out
        elif isinstance(other, (float, int)):
            def _backward():
                self.grad += 1 / other * out.grad
            out = Value(self.data / other, children=[self], _backward=_backward, _op='÷')
            return out
        
    def cross_entropy(logits, actual):
        maxVal = max([num.data for num in logits])

        exp = [(math.e**(num-maxVal)) for num in logits]

        count = sum([num for num in exp])

        prob = [val/count for val in exp]

        loss = prob[actual].log()*-1

        return loss

    def update_name(self, name):
        self.name = name
