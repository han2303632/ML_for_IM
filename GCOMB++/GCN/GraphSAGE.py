from GCN.Aggregator import Aggregator
import torch
import torch.nn as nn
import numpy
import random
from torch.autograd import Variable


import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, layer_infos):
        super(GraphSAGE, self).__init__()
        self.layer_infos = layer_infos
        self.aggregator1 = Aggregator(layer_infos[0])
        self.aggregator2 = Aggregator(layer_infos[1])
        self.aggregators = [self.aggregator1, self.aggregator2]
        self.embed_dim = layer_infos[-1].output_dim
        self.gcn = True


    def forward(self, features, adj_lists, edge_weight, nodes, mask):

        samples, support_size, num_valid_sample = self.sample(nodes, adj_lists, edge_weight)
        masked_feature = features.masked_fill(mask = mask, value=torch.tensor(-1))
        # masked_featu5re = self.features.masked_fill(mask = mask, value=torch.tensor(0))
        # row_index = torch.tensor([[i] for i in range(len(mask))])
        hidden = [masked_feature[sample] for sample in samples]

        for iter_idx, aggregator in enumerate(self.aggregators):
            next_hidden = []           
            for k in range(len(hidden) - 1):
                node = hidden[k+1]
                nbr = torch.reshape(hidden[k], (support_size[iter_idx + k+1], -1, aggregator.input_dim))
                new_vec = aggregator([node, nbr], num_valid_sample[iter_idx+k].cuda())
                next_hidden.append(new_vec)
            hidden = next_hidden
        return hidden[0]
        

    def sample(self, nodes, adj_lists, edge_weight):
        support_size = [len(nodes)]
        samples = [torch.tensor(nodes, dtype=torch.long)]
        _sample = numpy.random.choice
        num_valid_sample = []

        adj_lists[-1] = list()

        for k in range(len(self.layer_infos)):
            num_samples = self.layer_infos[k].num_samples
            node_nbr_sample = []
            neighbors_sample = []
            valid_sample = []
            support_size.append(support_size[k] * num_samples)
            
            for node in samples[k]:
                node = int(node)
                if len(adj_lists[node]) < num_samples:
                    sample_result = adj_lists[node] + [node]
                    # sample_result.append(node)
                    valid_sample.append([len(sample_result)])
                    node_nbr_sample = sample_result + [-1 for i in range(num_samples - len(adj_lists[node]) -1)]
                    if len(node_nbr_sample) < 5:
                        print(node)
                else:
                    sample_result = _sample(adj_lists[node], num_samples, replace=False)
                    valid_sample.append([len(sample_result)])
                    node_nbr_sample = sample_result
                    
                neighbors_sample.extend(node_nbr_sample)
            num_valid_sample.append(torch.tensor(valid_sample))
            samples.append(torch.tensor(neighbors_sample, dtype=torch.long))
            
        samples.reverse()
        num_valid_sample.reverse()
        support_size.reverse()

        return samples, support_size, num_valid_sample
        

