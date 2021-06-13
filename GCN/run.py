from Aggregator import Aggregator
from GraphSAGE import GraphSAGE
from collections import defaultdict, namedtuple
import torch.nn as nn
import torch
import numpy as np
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from torch.autograd import Variable


# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'input_dim',
     'output_dim' # the output (i.e., hidden) dimension
    ])

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds.t())

        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes + 1, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("GraphSAGE-master/cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    feat_data[-1,:]= [0 for i in range(num_feats)]

    adj_lists = defaultdict(set)
    with open("GraphSAGE-master/cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 2708
    feat_dim = 1433
    hidden_dim = 128
    feat_data, labels, adj_lists = load_cora()


    layer_infos = [SAGEInfo("node", None, 10, feat_dim, hidden_dim),
        SAGEInfo("node", None, 10, hidden_dim, hidden_dim)]
    sage = GraphSAGE(feat_data, num_nodes, layer_infos, 0, adj_lists)
    # sage = GraphSAGE(feat_data, num_nodes, feat_dim, layer_infos, adj_lists)
    graphsage = SupervisedGraphSage(7, sage)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(100):
        batch_nodes = train[:256]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data)
        # print(batch, loss.data[0])

    val_output = graphsage.forward(val) 
    print ("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print ("Average batch time:", np.mean(times))

if __name__ == "__main__":
    run()