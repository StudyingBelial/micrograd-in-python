import numpy as np

from helper_fucntions import unbroadcast

class Tensor:
    __array_priority__ = 1000

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

            if self.data.ndim == 1 and other.data.ndim == 1:
                self_grad = out.grad * other.data
                other_grad = out.grad * self.data
            elif self.data.ndim == 1 and other.data.ndim == 2:
                self_grad = out.grad @ other.data.T
                other_grad = np.outer(self.data, out.grad)
            else:
                def safe_T(arr):
                    return np.swapaxes(arr, -1, -2) if arr.ndim >= 2 else arr
                self_grad = out.grad @ safe_T(other.data)
                other_grad = safe_T(self.data) @ out.grad

            self.grad += unbroadcast(self_grad, self.data.shape)
            other.grad += unbroadcast(other_grad, other.data.shape)

        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))
        out._prev = {self}

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                shape_with_kept_dims = list(self.data.shape)
                axes = (axis,) if isinstance(axis, int) else axis
                for ax in axes:
                    shape_with_kept_dims[ax % self.data.ndim] = 1
                grad = np.reshape(grad, shape_with_kept_dims)

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
    
    def log(self, eps=1e-8):
        safe_data = np.clip(self.data, eps, None)
        out = Tensor(np.log(safe_data))
        out._prev = {self}
                
        def _backward():
            self.grad += (1 / self.data) * out.grad
            
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data)
        out._prev = {self}
        
        def _backward():
            self.grad -= out.grad
            
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
    
    def dropout(self, dropout_rate, training=True):
        if not training or dropout_rate == 0.0:
            return self
        
        mask = (np.random.rand(*self.data.shape) > dropout_rate).astype(self.data.dtype)
        
        scale = 1.0 / (1.0 - dropout_rate)
        mask = mask * scale
        
        out = Tensor(self.data * mask)
        out._prev = {self}
        
        def _backward():
            self.grad += mask * out.grad
        
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
        
        self.grad = np.ones_like(self.data) if grad is None else grad
        for node in reversed(topo):
            node._backward()