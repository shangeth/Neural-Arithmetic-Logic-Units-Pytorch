# Neural-Arithmetic-Logic-Units-Pytorch
This is the implementation of [(this paper)](https://arxiv.org/pdf/1808.00508v1.pdf) paper modules for Neural Accumulator and Neural Arithmetic Logic Unit using Pytorch.

## Neural Accumulator
![Neural Accumulator](https://cdn-images-1.medium.com/max/1600/1*vMYerlUvUP5gw4LDZv-aSg.png)
Neural Accumulator helps the network to explore different possible range of inputs and generalize it. Neural networks can't really do a good job in fitting arithmetic operations, Neural Accumulator makes the network take wide range of values and generalize even for extrapolation.

```python
class NeuralAccumulator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NeuralAccumulator, self).__init__()
        self.W1 = Parameter(torch.Tensor(out_dim, in_dim))
        self.W2 = Parameter(torch.Tensor(out_dim, in_dim))
        self.register_parameter('bias', None)
        
        self.W = Parameter(torch.tanh(self.W1) * torch.sigmoid(self.W2))
        
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        
    def forward(self, x):
        out = nn.functional.linear(x, self.W, self.bias)
        return out
```

### Neural Arithmetic Logic Unit
![](https://cdn-images-1.medium.com/max/1200/1*enYojyC0tBLv9BIJWkYVRg.png)
Although Neural Accumulator do a great job in arthmetics like +, - . It does not generalize multiplication, division, root, power, etc. as its just a combinations of +, - , 0 to the inputs to produce wide range.

Neural Arithmetic Logic Unit can generalise the other arithemetic operations as its a logarithmic and exponential function of Neural Accumulator.

```python
class NALU(nn.Module):
    def __init__(self, in_dim, out_dim, e=1e-5):
        super(NALU, self).__init__()
        self.e = e
        self.G = Parameter(torch.Tensor(out_dim, in_dim))
        self.W = Parameter(torch.Tensor(out_dim, in_dim))
        self.register_parameter('Gbias', None)
        self.register_parameter('Wbias', None)
        self.nac = NeuralAccumulator(in_dim, out_dim)
        
        nn.init.xavier_uniform_(self.G)
        nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        a = self.nac(x)
        g = torch.sigmoid(nn.functional.linear(x, self.G, self.Gbias))
        m = torch.exp(nn.functional.linear(torch.log(torch.abs(x) + self.e), self.W, self.Wbias))
        out = g*a + (1-g)*m
        return out
```


## Loss of models
### Addition

Iteration | MLP | Neural Accumulator
--- | --- | ---
 1000| 116.6307526  | 0(0 at 50th iter)


### Multiplication

Iteration | Neural Arithmetic Logic Unit | Neural Accumulator
--- | --- | ---
 10000| 0.0146152   | 152.5235901 

