import torch
from torch import nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

PATH = "./toy.pth"

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(10,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )

    def forward(self, x):
        x = self.stack(x)
        y = nn.functional.sigmoid(x)
        return y

model = MLP()
torch.save(model, PATH)

# you can load a PyTorch model from scratch
model = torch.load(PATH, weights_only=False)

traced : fx.GraphModule = fx.symbolic_trace(model)
graph = traced.graph

test_input = torch.randn([1,10])
out_golden = model(test_input)
out_actual = traced(test_input)
print(out_actual - out_golden)

print(traced.graph)
for n in traced.graph.nodes:
    print(n.name, n.op, n.target, n.args) # n.meta contains everything you need

ShapeProp(traced).propagate(test_input)

print("-------")
for x in traced.graph.nodes:
    tm = x.meta.get("tensor_meta")
    print(x.name, tm.shape, tm.dtype)
