import math

class Optimize:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
    
    def __repr__(self):
        return 'model shape: ' + str(self.model.parameters().shape)
    
    def zero_grad(self):
        self.model.zero_grad() 
    
class SGD(Optimize):
    def __init__(self, model, lr=1e-2):
        super().__init__(model, lr)

    def step(self):
        for p in self.model.parameters():
            p.data -= self.lr * p.grad

    
class Momentum(Optimize):
    def __init__(self, model, decay_rate=0.99, lr=1e-2):
        super().__init__(model, lr)
        self.momentum = [0] * self.model.parameters().shape[0]
        self.decay_rate = decay_rate
    
    def step(self):
        for i, p in enumerate(self.model.parameters()):
            self.momentum[i] = self.momentum[i] * self.decay_rate + (1.0-self.decay_rate) * p.grad
            p.data -= self.lr * self.momentum[i]

class Adagrad(Optimize):
    def __init__(self, model, lr=1e-2):
        super().__init__(model, lr)
        self.m = [0] * self.model.parameters().shape[0]
    
    def step(self):
        for i, p in enumerate(self.model.parameters()):
            self.m[i] = self.m[i] + (p.grad)**2
            p.data -= self.lr * p.grad / (math.sqrt(self.m[i])+1e-8)

class RMSprop(Optimize):
    def __init__(self, model, decay_rate=0.99, lr=0.1):
        super().__init__(model, lr)
        self.decay_rate = decay_rate
        self.m = [0] * self.model.parameters().shape[0]

    def step(self):
        for i, p in enumerate(self.model.parameters()):
            self.m[i] = self.decay_rate*self.m[i] + (1-self.decay_rate)*(p.grad)**2
            p.data -= self.lr * p.grad / (math.sqrt(self.m[i]) + 1e-8)

class Adam(Optimize):
    def __init__(self, model, decay_rate_1=0.99, decay_rate_2=0.9, lr=1e-2):
        super().__init__(model, lr)
        self.decay_rate_1 = decay_rate_1
        self.decay_rate_2 = decay_rate_2
        self.v = [0] * self.model.parameters().shape[0]
        self.s = [0] * self.model.parameters().shape[0]
        self.iter = 0

    def step(self):
        self.iter += 1
        for i, p in enumerate(self.model.parameters()):
            self.v[i] = self.decay_rate_1 * self.v[i] + (1-self.decay_rate_1)*p.grad
            self.s[i] = self.decay_rate_2 * self.s[i] + (1-self.decay_rate_2)*((p.grad)**2)
            v = self.v[i] / (1 - ((self.decay_rate_1)**self.iter))
            s = self.s[i] / (1 - ((self.decay_rate_2)**self.iter))
            p.data -= self.lr * (v / (math.sqrt(s)+1e-8))

class AdamW(Optimize):
    def __init__(self, model, decay_rate_1=0.99, decay_rate_2=0.9, lr=1e-2, weight_decay=0.01):
        super().__init__(model, lr)
        self.decay_rate_1 = decay_rate_1
        self.decay_rate_2 = decay_rate_2
        self.weight_decay = weight_decay
        self.v = [0] * self.model.parameters().shape[0]
        self.s = [0] * self.model.parameters().shape[0]
        self.iter = 0

    def step(self):
        self.iter += 1
        for i, p in enumerate(self.model.parameters()):
            self.v[i] = self.decay_rate_1 * self.v[i] + (1-self.decay_rate_1)*p.grad
            self.s[i] = self.decay_rate_2 * self.s[i] + (1-self.decay_rate_2)*((p.grad)**2)
            v = self.v[i] / (1 - ((self.decay_rate_1)**self.iter))
            s = self.s[i] / (1 - ((self.decay_rate_2)**self.iter))
            p.data -= self.lr * (v / (math.sqrt(s)+1e-8) + self.weight_decay * p.data)