import torch
import torch.nn as nn
from torch.nn.modules.container import ModuleList
from itertools import repeat
import torch.nn.functional as F

# class Config(object):
#     def __init__(self):
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")
#         self.max_len = 1200
#         self.embedding_size = 32
#         self.Kernel_size = [(3,64),(5,64),(7,64),(9,64),(11,128),(13,128),(15,128)]
#         sum_dim = 0
#         for K in self.Kernel_size:
#             sum_dim += K[1]
#         self.cnn_output_size = sum_dim

# config = Config()


# class MLP(nn.Module):
#     def __init__(self,input_size, mlp_layers=[128,64,32],
#                     ):
#         super(MLP,self).__init__()
#         mlp_layers.insert(0, input_size)
#         self.linears = nn.ModuleList([nn.Linear(mlp_layers[index],mlp_layers[index+1]) for index in range(len(mlp_layers)-2)])


# def mlp_layer(input_layer,
#               mlp_layers=[128,64,32],
#               activation=nn.ReLU,
#               last_layer_activation=None,
#               use_dropout=False,
#               drop_prob=0.2,
#               ):
#     """
#     define mlp layer graph
#     Args:
#         input_layer: 
#         mlp_layers:

#     """

class CNN(nn.Module):
    """
    textCNN
    input_tensor_shape: [batch_size, max_len, embedding_size]
    output_tensor_shape: [batch_size, cnn_output_size]
    """
    def __init__(self, config):
        super(CNN, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(1,K[1],(K[0],config.embedding_size)) for K in config.Kernel_size])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,x):
        # print(x.shape)
        out = x.unsqueeze(1)
        # print(out.shape)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        return out        

class Expert(nn.Module):
    """
    input_tensor_shape: [batch_size, max_len, embedding_size]
    output_tensor_shape: [batch_size, expert_output_size]
    """
    def __init__(self, config):
        super(Expert, self).__init__()
        self.cnn = CNN(config)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(config.cnn_output_size, config.expert_output_size)

    def forward(self,x):
        out = self.cnn(x)
        out = self.relu(self.fc(out))
        out = self.dropout(out)
        return out

class Gate(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.fc1 = nn.Linear(config.gate_input_size, config.gate_hidden_size)
        self.fc2 = nn.Linear(config.gate_hidden_size, config.gate_out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out

class Tower(nn.Module):
    def __init__(self, config, classes_nums):
        super(Tower, self).__init__()
        if config.tower_version == 0:
            self.fc1 = nn.Linear(config.expert_output_size, config.tower_hidden_size)
        else:
            self.fc1 = nn.Linear(config.expert_output_size*config.num_experts, config.tower_hidden_size)
        self.fc2 = nn.Linear(config.tower_hidden_size, classes_nums)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class SingleTask_g_pos(nn.Module):
    def __init__(self,config):
        super(SingleTask_g_pos,self).__init__()
        self.W = nn.Embedding(config.vocab_size, config.embedding_size)
        self.bottom = Expert(config)
        self.tower = Tower(config,4)
    
    def forward(self,x):
        x = self.W(x)
        out = self.bottom(x)
        out = self.tower(out)
        return out

class SingleTask_g_neg(nn.Module):
    def __init__(self,config):
        super(SingleTask_g_neg,self).__init__()
        self.W = nn.Embedding(config.vocab_size, config.embedding_size)
        self.bottom = Expert(config)
        self.tower = Tower(config,5)
    
    def forward(self,x):
        x = self.W(x)
        out = self.bottom(x)
        out = self.tower(out)
        return out

class ShareBottom(nn.Module):
    def __init__(self,config):
        super(ShareBottom, self).__init__()
        self.W = nn.Embedding(config.vocab_size, config.embedding_size)
        self.bottom = Expert(config)
        self.towerA = Tower(config,4)
        self.towerB = Tower(config,5)

    def forward(self,x):
        x = self.W(x)
        out = self.bottom(x)
        outA = self.towerA(out)
        outB = self.towerB(out)
        return [outA, outB]

class CustomizedShare(nn.Module):
    def __init__(self,config):
        super(CustomizedShare, self).__init__()
        self.W = nn.Embedding(config.vocab_size, config.embedding_size)
        self.bottomA = Expert(config)
        self.bottomS = Expert(config)
        self.bottomB = Expert(config)
        self.towerA = Tower(config, 4)
        self.towerB = Tower(config, 5)
    
    def forward(self, x):
        x = self.W(x)
        bottomA_out = self.bottomA(x)
        bottomS_out = self.bottomS(x)
        bottomB_out = self.bottomB(x)
        outA = self.towerA(bottomA_out+bottomS_out)
        outB = self.towerB(bottomB_out+bottomS_out)
        return [outA, outB]#, [bottomA_out, bottomS_out, bottomB_out]


class AttentionShare(nn.Module):
    def __init__(self,config):
        super(AttentionShare, self).__init__()
        self.W = nn.Embedding(config.vocab_size, config.embedding_size)
        self.bottomA = Expert(config)
        self.bottomS = Expert(config)
        self.bottomB = Expert(config)
        self.gateA = Gate(config)
        self.gateB = Gate(config)
        self.towerA = Tower(config, 4)
        self.towerB = Tower(config, 5)
        self.expert_output_size = config.expert_output_size
    
    def forward(self, x):
        x = self.W(x)
        bottomA_out = self.bottomA(x)
        bottomS_out = self.bottomS(x)
        bottomB_out = self.bottomB(x)
        # print("out_A shape: {}".format(bottomA_out.shape))
        # print("out_S shape: {}".format(bottomS_out.shape))
        # print("out_B shape: {}".format(bottomB_out.shape))
        gate_inputA = torch.cat([bottomA_out,bottomS_out],1)
        gate_inputB = torch.cat([bottomB_out,bottomS_out],1)
        # print("gate_A_input shape: {}".format(gate_inputA.shape))
        # print("gate_B_input shape: {}".format(gate_inputB.shape))
        gateA_out = self.gateA(gate_inputA)
        gateB_out = self.gateB(gate_inputB)
        # print("gate_A_output shape: {}".format(gateA_out.shape))
        # print("gate_B_output shape: {}".format(gateB_out.shape))

        outA = self.towerA(gateA_out[:,0].unsqueeze(1).expand(-1,self.expert_output_size) * bottomA_out\
            + gateA_out[:,1].unsqueeze(1).expand(-1,self.expert_output_size) * bottomS_out)
        outB = self.towerB(gateB_out[:,0].unsqueeze(1).expand(-1,self.expert_output_size) * bottomB_out\
            + gateB_out[:,1].unsqueeze(1).expand(-1,self.expert_output_size) * bottomS_out)
        return [outA, outB]

# class MMoE(nn.Module):
#     def __init__(self,config):
#         super(MMoE, self).__init__()
#         self.W = nn.Embedding(config.vocab_size, config.embedding_size)

#         self.experts_out = config.expert_output_size
#         self.softmax = nn.Softmax(dim=1)

#         self.experts = nn.ModuleList([Expert(config) for i in range(config.num_experts)])
#         self.gates = nn.ModuleList([Gate(config) for i in range(self.tasks)])
#         self.towers = nn.ModuleList([Tower(config) for i in range(config.tasks)])
    
#     def forward(self, x):
#         x = self.W(x)
#         experts_o = [e(x) for e in self.experts]
#         experts_o_tensor = torch.stack(experts_o)

#         gates_o = [self.softmax(x @ g) for g in self.w_gates]

#         tower_input = [g.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o_tensor for g in gates_o]
#         tower_input = [torch.sum(ti, dim=0) for ti in tower_input]

#         final_output = [t(ti) for t, ti in zip(self.towers, tower_input)]
#         return 

class SpatialDropout(nn.Module):
    """
    空间dropout，即在指定轴方向上进行dropout，常用于Embedding层和CNN层后
    如对于(batch, timesteps, embedding)的输入，若沿着axis=1则可对embedding的若干channel进行整体dropout
    若沿着axis=2则可对某些token进行整体dropout
    """
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])   # 默认沿着中间所有的shape
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)


