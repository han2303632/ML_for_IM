import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import torch.nn.functional as F

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, feat_dim, embed_dim, adj_lists, base_model=None, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.adj_lists = adj_lists
        self.base = base_model
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, nodes, num_sample=5):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        to_neighs =  [self.adj_lists[int(node)] for node in nodes]
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))

        # find all nodes in one hop between a node(including itself if using gcn)
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # construct node-idx map
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)

        self_feats = self.features(torch.LongTensor(nodes))
        combined = torch.cat([self_feats, to_feats], dim=1)
        combined = F.relu(self.weight.mm(combined.t()))

        return combined
        '''


        feature = self.input_features[:-1]
        # for iter_idx, aggregator in enumerate(self.aggregators):
        #     num_sample = self.layer_infos[iter_idx].num_samples
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
        '''
