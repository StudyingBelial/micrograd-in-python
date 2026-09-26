import numpy as np

class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
        
    def zero_grad(self):
        params = self.parameters
        for p in params:
            p.grad = np.zeros_like(p.data)

class RMSProp(Optimizer):
    def __init__(self, parameters, lr=0.001, dr=0.9, eps=1e-8):
        super().__init__(parameters, lr)
        self.dr = dr
        self.eps = eps
        self.state = {p: 0.0 for p in self.parameters}
        
    def step(self):
        for p in self.parameters:
            self.state[p] = (self.state[p] * self.dr) + ((1 - self.dr) * (p.grad **2))
            p.data -= (self.lr / ((self.state[p] ** 0.5) + self.eps)) * p.grad

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.001, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = {p : 0.0 for p in self.parameters}
    
    def step(self):
        for p in self.parameters:
            self.velocity[p] = self.momentum * self.velocity[p] - self.lr * p.grad
            p.data += self.velocity[p]
            
class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, direction_decay=0.9, steepness_decay=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.direction_decay = direction_decay
        self.steepness_decay = steepness_decay
        self.eps = eps
        self.step_count = 1
        self.steepness = {p : 0.0 for p in self.parameters}
        self.direction = {p : 0.0 for p in self.parameters}
        
    def step(self):
        for p in self.parameters:
            self.direction[p] = (self.direction_decay * self.direction[p]) + ((1 - self.direction_decay) * p.grad)
            self.steepness[p] = (self.steepness_decay * self.steepness[p]) + ((1 - self.steepness_decay) * (p.grad ** 2))
            
            corrected_direction = self.direction[p] / (1 - (self.direction_decay ** self.step_count))
            corrected_steepness = self.steepness[p] / (1 - (self.steepness_decay ** self.step_count))
            
            p.data -= (self.lr / ((corrected_steepness ** 0.5) + self.eps)) * corrected_direction
                
        self.step_count += 1