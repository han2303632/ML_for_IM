from Aggregator import Aggregator
import torch
import torch.nn as nn
import numpy
import random
from torch.autograd import Variable


import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, feature, num_node, layer_infos, i, adj_lists):
        super(GraphSAGE, self).__init__()
        self.all_nodes = [i for i in range(num_node)]
        # self.input_features = torch.FloatTensor(feature)
        self.features = torch.FloatTensor(feature)
        self.layer_infos = layer_infos
        self.i = i
        self.neighbors = adj_lists
        self.aggregator1 = Aggregator(layer_infos[0])
        self.aggregator2 = Aggregator(layer_infos[1])
        # self.aggregators = [Aggregator(layer_infos[0]), Aggregator(layer_infos[1])]
        self.aggregators = [self.aggregator1, self.aggregator2]
        self.embed_dim = layer_infos[-1].output_dim
        self.gcn = True


        self.weight = nn.Parameter(
                        torch.FloatTensor(layer_infos[i].output_dim, 2 * layer_infos[i].input_dim))
        nn.init.xavier_uniform_(self.weight)



    def forward(self, nodes):

        samples, support_size, num_valid_sample = self.sample(nodes)
        hidden = [self.features[sample] for sample in samples]


        # aggreagtor iteration
        # for  in range(len(layer_info)):
        #      aggregator initialization
        for iter_idx, aggregator in enumerate(self.aggregators):
            next_hidden = []           
            for k in range(len(hidden) - 1):
                node = hidden[k+1]
                nbr = torch.reshape(hidden[k], (support_size[iter_idx + k+1], aggregator.num_sample, aggregator.input_dim))
                new_vec = aggregator([node, nbr], num_valid_sample[iter_idx+k])
                next_hidden.append(new_vec)
            hidden = next_hidden
        return hidden[0]
        '''
        feature = self.features
        for iter_idx, aggregator in enumerate(self.aggregators):
            num_sample = self.layer_infos[iter_idx].num_samples
            _set = set
            if not num_sample is None:
                _sample = random.sample
                samp_neighs = [_set(_sample(to_neigh, 
                                num_sample,
                                )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in [self.neighbors[int(node)] for node in self.all_nodes]] 
                                # )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in self.neighbors] 

            # samp_neighs = [samp_neigh + set([self.all_nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
            unique_nodes_list = list(set.union(*samp_neighs))

            # find all nodes in one hop between a node(including itself if using gcn)
            unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
            # construct node-idx map
            mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
            column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
            row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
            mask[row_indices, column_indices] = 1
            num_neigh = mask.sum(1, keepdim=True)
            mask = mask.div(num_neigh)
            embed_matrix = feature[torch.LongTensor(unique_nodes_list)]
            to_feats = mask.mm(embed_matrix)
            feature = aggregator([feature, to_feats])

        return feature[nodes]
        return to_feats
        '''
        
        '''
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        num_sample = 5# self.layer_infos[self.i].num_sample
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        # if self.gcn:
        #     samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))

        # find all nodes in one hop between a node(including itself if using gcn)
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # construct node-idx map
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # if self.cuda:
        #     mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        # if self.cuda:
        #     embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        # else:
        #     embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)


        self_feats = self.features(torch.LongTensor(nodes))
        combined = torch.cat([self_feats, to_feats], dim=1)
        combined = F.relu(self.weight.mm(combined.t()))

        return combined

        '''


    def sample(self, nodes):
        support_size = [len(nodes)]
        samples = [torch.tensor(nodes, dtype=torch.long)]
        _sample = numpy.random.choice
        num_valid_sample = []

        self.neighbors[-1] = set()

        for k in range(len(self.layer_infos)):
            num_samples = self.layer_infos[k].num_samples
            node_nbr_sample = []
            neighbors_sample = []
            valid_sample = []
            support_size.append(support_size[k] * num_samples)
            
            for node in samples[k]:
                node = int(node)
                if len(self.neighbors[node]) < num_samples:
                    sample_result = list(self.neighbors[node]) + [node]
                    # sample_result.append(node)
                    valid_sample.append([len(sample_result)])
                    node_nbr_sample = sample_result + [-1 for i in range(num_samples - len(self.neighbors[node]) -1)]
                else:
                    sample_result = _sample(list(self.neighbors[node]), num_samples, replace=False)
                    valid_sample.append([len(sample_result)])
                    node_nbr_sample = sample_result
                    
                neighbors_sample.extend(node_nbr_sample)
            num_valid_sample.append(torch.tensor(valid_sample))
            samples.append(torch.tensor(neighbors_sample, dtype=torch.long))
            
        samples.reverse()
        num_valid_sample.reverse()
        support_size.reverse()

        return samples, support_size, num_valid_sample
        


