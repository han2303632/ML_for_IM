import torch.nn as nn
from torch import mean
import torch


class Aggregator(nn.Module):
    def __init__(self, layer_info):
        super(Aggregator, self).__init__()
        self.num_sample = layer_info.num_samples
        self.input_dim = layer_info.input_dim
        self.output_dim = layer_info.output_dim
        self.lin1 = nn.Linear(2 * self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.lin1.weight)

    def forward(self, inputs, num_valid_sample = None):
        # neighbor sampling
        sampling_neighbors = []
        node_vec, nbr_vec = inputs
        mean_nbr_vec = torch.div(torch.sum(nbr_vec,  dim=1), num_valid_sample)

        # out_vec = self.lin1(torch.cat((node_vec, mean_nbr_vec), dim =1))
        out_vec = self.lin1(torch.cat((node_vec, mean_nbr_vec), dim =1))

        return nn.ReLU(inplace=True)(out_vec)








        
            
            



