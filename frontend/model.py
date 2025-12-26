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
        z = nn.functional.sigmoid(x*-1)
        return y+z

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
    mod = "-"
    if  x.op == "call_module":
        mod = traced.get_submodule(x.target)
    print(x.name, tm.shape, tm.dtype, mod)

print("*****")
print([x for x,y in traced.named_parameters()])

class IRNode():
    def __init__(self): 
        self.name = ""
        self.obj_type = ""
        self.dtype = ""
        self.shape = ""
        self.op_name = ""
        self.inputs = []
        self.parameters = []
        self.output = ""
    def print(self):
        print(f"""
            name = {self.name}, \n
            obj_type = {self.obj_type}, \n
            dtype = {self.dtype}, \n
            shape = {self.shape}, \n
            op_name = {self.op_name}, \n
            inputs = {self.inputs}, \n
            parameters = {self.parameters}, \n
            output = {self.output}, \n """
        )
        
IR_nodes = [] # each entry is an IRNode (name, op_type, op_name, type, [inputs], [parameters], output)

def transform_graph(traced, IR_nodes):
    for n in traced.graph.nodes:
        tm = n.meta.get("tensor_meta")
        new_node = IRNode()
        new_node.name = n.name
        new_node.obj_type = n.op
        new_node.dtype = tm.dtype
        new_node.shape = torch.tensor(tm.shape).tolist()

        if n.op == "placeholder":
            new_node.op_name = "input"
        if n.op == "call_method":
            new_node.op_name = n.target
        if n.op == "output":
            new_node.op_name = n.target
        if n.op == "call_module":
            mod = traced.get_submodule(n.target)
            if isinstance(mod, nn.Linear):
                new_node.op_name = "Linear"
            elif isinstance(mod, nn.ReLU):
                new_node.op_name = "ReLU"
            else:
                new_node.op_name = ""
            tmp_parameters = [[x[0],torch.tensor(x[1].shape).tolist()] for x in mod.named_parameters()]
            new_node.parameters = tmp_parameters
            
        new_node.inputs = list(n.args)
        new_node.output = n.name
        


        IR_nodes.append(new_node)
    return 

transform_graph(traced,IR_nodes)

print("IR_nodes:")
for n in IR_nodes:
    n.print()