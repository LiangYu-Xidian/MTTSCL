from tkinter import OptionMenu
import torch
import torch.nn as nn

class Config_h(object):
    def __init__(self):

        # global config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 1000
        self.embedding_size = 32
        self.vocab_size = 25
        self.batch_size = 32
        self.lr = 1e-3
        self.N_EPOCHS = 1

        self.classes_g_pos = ["C","CM","E","CW"]
        self.classes_g_neg = ["C","CM","OM","E", "P"]

        # cnn config
        self.Kernel_size = [(3,32),(5,32),(7,32),(9,32),(11,32),(13,32),(15,32)]
        sum_dim = 0
        for K in self.Kernel_size:
            sum_dim += K[1]
        self.cnn_output_size = sum_dim

        # expert config
        self.expert_output_size = 128

        # gate config
        self.gate_hidden_size = 32
        self.gate_out_size = 2
        self.gate_input_size = self.expert_output_size * 2


        # tower config
            # tower_version: 
            # tower_version=0, tower input as sum([expert_n] * gate)
            # tower_version=1, tower input as concat([expert_n] * gate)
        self.tower_version = 0 
        self.tower_hidden_size = 32
        self.tower_output_size = 4

        # mmoe config
        self.num_experts = 3
        self.tasks = 2


class Config_s(object):
    def __init__(self):

        # global config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = 1000
        self.embedding_size = 32
        self.vocab_size = 25
        self.batch_size = 32
        self.lr = 1e-3
        self.N_EPOCHS = 5

        self.classes_g_pos = ["C","CM","E","CW"]
        self.classes_g_neg = ["C","CM","OM","E", "P"]

        # cnn config
        # self.Kernel_size = [(3,32),(5,32),(7,32),(9,32),(11,32),(13,32),(15,32)]
        self.Kernel_size = [(3,32),(5,32),(7,32),(9,32),(11,32),(13,32),(15,32)]
        sum_dim = 0
        for K in self.Kernel_size:
            sum_dim += K[1]
        self.cnn_output_size = sum_dim

        # expert config
        self.expert_output_size = 32

        # gate config
        self.gate_hidden_size = 16
        self.gate_out_size = 2
        self.gate_input_size = self.expert_output_size * 2


        # tower config
            # tower_version: 
            # tower_version=0, tower input as sum([expert_n] * gate)
            # tower_version=1, tower input as concat([expert_n] * gate)
        self.tower_version = 0 
        self.tower_hidden_size = 16
        self.tower_output_size = 4

        # mmoe config
        self.num_experts = 3
        self.tasks = 2

config = Config_h()