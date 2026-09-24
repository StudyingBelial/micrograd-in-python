class ReLU():
    def __call__(self, x):
        return x.relu()

    def parameters(self):
        return []
    
class Sigmoid():
    def __call__(self, x):
        return x.sigmoid()

    def parameters(self):
        return []
    
class Tanh():
    def __call__(self, x):
        return x.tanh()
    
    def parameters(self):
        return []
    
class Softmax():
    def __call__(self, x):
        return x.softmax()
        
    def parameters(self):
        return []
    
class LeakyReLU():
    def __call__(self, x, multiplier=0.01):
        return x.leakyrelu(multiplier)
    
    def parameters(self):
        return []