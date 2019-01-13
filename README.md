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


## Loss of models
### Addition

Iteration | MLP | Neural Accumulator
--- | --- | ---
 1000| 116.6307526  | 0(0 at 50th iter)


### Multiplication

Iteration | Neural Arithmetic Logic Unit | Neural Accumulator
--- | --- | ---
 10000| 0.0146152   | 152.5235901 

