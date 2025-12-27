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

class FXNode():
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

class IRNode():
    def __init__(self): 
        self.id = ""
        self.args = []
        self.params = []
        self.result = ""
        self.out_shape = ""
        self.out_type = ""
        self.op_type = ""
    def print(self):
        print(f"""
            id = {self.id}
            args = {self.args}
            params = {self.params}
            result = {self.result}
            out_shape = {self.out_shape}
            out_type = {self.out_type}
            op_type = {self.op_type} """
        )
        
FX_nodes = [] # each entry is an FXNode (name, op_type, op_name, type, [inputs], [parameters], output)

def transform_graph(traced, FX_nodes):
    for n in traced.graph.nodes:

        assert(n.meta.get("tensor_meta") != None)
        tm = n.meta.get("tensor_meta")
        new_node = FXNode()
        new_node.name = n.name
        new_node.obj_type = n.op
        new_node.dtype = tm.dtype
        new_node.shape = torch.tensor(tm.shape).tolist()
        new_node.target = n.target
        new_node.output = n.name
        new_node.inputs = [str(x) for x in list(n.args)]

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
            new_node.op_name = "output"
            new_node.output = "output"
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
        FX_nodes.append(new_node)
    return 

transform_graph(traced,FX_nodes)

# assign IDs:
print("FX nodes w/o ID:")
node_ids = dict()
arg_counter = 0
var_counter = 0
param_counter = 0
for n in FX_nodes:
    assert(n.name not in node_ids.keys())
    if(n.obj_type == "placeholder"):
        node_ids[n.name] = "arg_" + str(arg_counter)
        arg_counter += 1
    if(n.obj_type == "call_module"):
        node_ids[n.name] = "%" + str(var_counter)
        var_counter += 1
    if(n.obj_type == "call_method"):
        node_ids[n.name] = "%" + str(var_counter)
        var_counter += 1
    if(n.obj_type == "output"):
        node_ids[n.name] = "*"
    n.print()

for name,_ in traced.named_parameters():
    node_ids[name] = "param_" + str(param_counter)
    param_counter += 1


print("Unifying names to IDs:")
# change names to IDs:

print(node_ids)
IR_nodes = []
for n in FX_nodes:
    assert(n.name in node_ids.keys())
    new_ir_node = IRNode()
    new_ir_node.id = node_ids[n.name]
    for inp in n.inputs:
        assert(inp in node_ids.keys())
        new_ir_node.args.append(node_ids[inp])
    for pname, pshape in n.parameters:
        assert(pname in node_ids.keys())
        new_ir_node.params.append((node_ids[pname], pshape))
    new_ir_node.result = node_ids[n.output]
    new_ir_node.out_shape = n.shape
    new_ir_node.out_type = n.dtype
    new_ir_node.op_type = n.op_name
    if n.obj_type == "output":
        new_ir_node.id = None
        new_ir_node.params = None
        new_ir_node.result = None
        new_ir_node.out_shape = None
        new_ir_node.op_type = "return"
    IR_nodes.append(new_ir_node)
    new_ir_node.print()


