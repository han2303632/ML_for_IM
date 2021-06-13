import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
from GCN.GraphSAGE import GraphSAGE
from  torch.optim.lr_scheduler import ExponentialLR

class Qfunction(nn.Module):
    def __init__(self, features, adj_lists, layer_infos, lr, graph_size, lr_gamma=0.95):
        super(Qfunction, self).__init__()


        self.sage = GraphSAGE(features, layer_infos, adj_lists)
        self.embed_dim = self.sage.embed_dim
        self.graph_size = graph_size
        
        self.lin1 = nn.Linear(self.embed_dim, self.embed_dim + 1, bias=False)
        self.lin2 = nn.Linear(self.embed_dim, self.embed_dim + 1, bias=False)
        self.lin3 = nn.Linear(self.embed_dim, self.embed_dim + 1, bias=False)
        self.lin4 = nn.Linear(2 * (self.embed_dim + 1) + self.embed_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.lin3.weight)
        nn.init.xavier_uniform_(self.lin4.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.zero_pad = torch.zeros(1, self.embed_dim).cuda()
        self.cuda(device=0)

    def forward(self, seeds_idx_pad, seeds_idx_num, candidates_idx, all_nodes, mask, batch_size=None):
        embedding = self.sage(all_nodes, mask)
        embedding_pad = torch.cat((embedding, self.zero_pad), dim=0)

        seeds_emb = embedding_pad[seeds_idx_pad]
        
        seeds_mean = nn.Sigmoid()(self.lin1(torch.sum(seeds_emb, dim=1).div(seeds_idx_num)))
        global_mean = nn.Sigmoid()(self.lin2(torch.mean(embedding, dim=0).reshape(1, self.embed_dim)))

        if batch_size is not None:
            seeds_mean = torch.cat([seeds_mean for i in range(batch_size)], dim=0)
            global_mean = torch.cat([global_mean for i in range(batch_size)], dim=0)

        candi_emb = nn.Sigmoid()(self.lin3(embedding[candidates_idx]))
        
        
        candidates_size = len(candidates_idx)
        cur_reward_frac = torch.sum(torch.mul(candi_emb, seeds_mean), dim=1)

        # future_reward_frac = 

        # Cat = torch.cat((candi_emb, seeds_mean, global_mean), dim=1)
        Q = (1 - torch.sum(torch.mul(candi_emb, seeds_mean), dim=1)) * self.graph_size
        # Q = nn.Sigmoid()(self.lin2(Cat).reshape(candidates_size)) * len(all_nodes)



        # origin
        # Cat = torch.cat((self.lin1(candidates_max_mat), self.lin2(seeds_max_mat), self.lin3(candidates_emb)), dim = 1)
        # Q = self.lin4(Cat).reshape(batch_size)
  

        return Q










        

