import numpy as np

from helper_fucntions import unbroadcast

class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float64)
        self._prev = set()
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        
    def transpose(self):
        out = Tensor(np.swapaxes(self.data, -1, -2))
        out._prev = {self}

        def _backward():
            self.grad += np.swapaxes(out.grad, -1, -2)

        out._backward = _backward
        return out
    
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape))
        out._prev = {self}

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            self_grad = out.grad @ np.swapaxes(other.data, -1, -2)
            other_grad = np.swapaxes(self.data, -1, -2) @ out.grad

            self.grad += unbroadcast(self_grad, self.data.shape)
            other.grad += unbroadcast(other_grad, other.data.shape)

        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward():
            grad = out.grad

            if axis is not None:
                if not keepdims:
                    axes = (axis,) if isinstance(axis, int) else axis
                    for ax in sorted(axes):
                        grad = np.expand_dims(grad, ax)

            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        divisor = self.data.size if axis is None else np.prod(
            np.asarray(self.data.shape)[list(axis)] 
            if isinstance(axis, tuple)
            else self.data.shape[axis]
        )

        out = self.sum(axis=axis, keepdims=keepdims) / divisor
        return out
        
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward 
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data)
        out._prev = {self, other}

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad -= unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data)
        out._prev = {self, other}
        
        def _backward():
            self.grad += unbroadcast((1/ other.data) * out.grad, self.data.shape)
            other.grad +=  unbroadcast((-self.data / other.data ** 2) * out.grad, other.data.shape)
            
        out._backward = _backward
        return out
            
    
    def __pow__(self, other):
        out = Tensor(self.data ** other)
        out._prev = {self}
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
            
        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data))
        out._prev = {self}
                
        def _backward():
            self.grad += (1 / self.data) * out.grad
            
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Tensor(other) - self if not isinstance(other, Tensor) else other - self
    
    def __rtruediv__(self, other):
        return Tensor(other) / self if not isinstance(other, Tensor) else other / self

    def relu(self):
        out = Tensor(np.maximum(0, self.data))
        out._prev = {self}

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
    
    def leakyrelu(self, multiplier):
        out = Tensor(np.maximum(self.data * multiplier, self.data))
        out._prev = {self}
        
        def _backward():
            self.grad += np.where(self.data > 0, 1, multiplier) * out.grad
        
        out._backward = _backward
        return out

    def sigmoid(self):
        out = Tensor(1.0 / (1.0 + np.exp(-self.data)))
        out._prev = {self}

        def _backward():
            self.grad += (out.data * (1.0 - out.data)) * out.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        out = Tensor((2 / (1 + np.exp(-2 * self.data))) - 1)
        out._prev = {self}
                
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def softmax(self):
        exp = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out = Tensor(exp/ np.sum(exp, axis=-1, keepdims=True))
        out._prev = {self}
                
        def _backward():
            self.grad += out.data * (out.grad - np.sum(out.grad * out.data, axis=-1, keepdims=True))
        
        out._backward = _backward
        return out

    def backward(self, grad=None):
        topo = []
        visited = set()
    
        def build(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)
        build(self)
        
        self.grad = 1.0 if grad is None  else grad
        for node in reversed(topo):
            node._backward()