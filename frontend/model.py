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
assert(torch.max(out_actual - out_golden ).item() < 1e-9) #checking if the golden model and traced graph output the same values

print(traced.graph)
for n in traced.graph.nodes:
    print(n.name, n.op, n.target, n.args) # n.meta contains everything you need

ShapeProp(traced).propagate(test_input) #adding shapes to each tensor's meta data

class IRNode():
    def __init__(self): 
        self.name = ""
        self.obj_type = ""
        self.dtype = ""
        self.shape = ""
        self.op_name = ""
        self.target = ""
        self.output = ""
        self.inputs = []
        self.parameters = []
    def print(self):
        print(f"""
            name = {self.name}
            obj_type = {self.obj_type}
            dtype = {self.dtype}
            shape = {self.shape}
            op_name = {self.op_name}
            target = {self.target}
            output = {self.output}
            inputs = {self.inputs}
            parameters = {self.parameters} """
        )
        
IR_nodes = [] # each entry is an IRNode (name, op_type, op_name, type, [inputs], [parameters], output)

def transform_graph(traced, IR_nodes):
    for n in traced.graph.nodes:

        assert(n.meta.get("tensor_meta") != None)
        tm = n.meta.get("tensor_meta")
        new_node = IRNode()
        new_node.name = n.name
        new_node.obj_type = n.op
        new_node.dtype = tm.dtype
        new_node.shape = torch.tensor(tm.shape).tolist()
        new_node.target = n.target
        new_node.output = n.name
        new_node.inputs = list(n.args)

        # searching for op_name and parameters:
        if n.op == "placeholder":
            new_node.op_name = "input"
        if n.op == "call_method":
            if n.target == "sigmoid":
                new_node.op_name = n.target
                print(type(n.target))
            else:
                new_node.op_name = ""
        if n.op == "call_function":
            new_node.op_name = ""
            # new_node.op_name = n.target
        if n.op == "output":
            new_node.op_name = "return"
        if n.op == "call_module":
            mod = traced.get_submodule(n.target)
            if isinstance(mod, nn.Linear):
                new_node.op_name = "linear"
            elif isinstance(mod, nn.ReLU):
                new_node.op_name = "relu"
            else:
                new_node.op_name = ""
            tmp_parameters = [[new_node.target+"."+x[0],torch.tensor(x[1].shape).tolist()] for x in mod.named_parameters()]
            new_node.parameters = tmp_parameters

        assert(new_node.op_name != "")
        IR_nodes.append(new_node)
    return 

transform_graph(traced,IR_nodes)

print("IR_nodes:")
for n in IR_nodes:
    n.print()