from Tensor import Tensor
import numpy as np

class Layer:
    def __init__(self, nin, nout):
        self.W = Tensor(np.random.randn(nin, nout) * np.sqrt(2.0 / nin))
        self.B = Tensor(np.zeros((1, nout)))

    def __call__(self, x):
        return x @ self.W + self.B
        
    def parameters(self):
        return [self.W, self.B]
    
class DropOut:
    def __init__(self, dropout_rate=0.0):
        self.dropout_rate = dropout_rate
        self.training = True
    
    def __call__(self, x):
        return x.dropout(training=self.training, dropout_rate=self.dropout_rate)
        
    def parameters(self):
        return []

class Base:
    def __init__(self):
        pass

    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params