import torch
import torch.nn as nn
import torch.optim as optim
from math import log
from torch.nn import init
from GCN.GraphSAGE import GraphSAGE
from  torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F

class Qfunction(nn.Module):
    def __init__(self, layer_infos, graph_size, lr, lr_gamma=0.95):
        super(Qfunction, self).__init__()


        self.sage = GraphSAGE(layer_infos)
        self.embed_dim = self.sage.embed_dim
        self.graph_size = graph_size
        
        self.lin1 = nn.Linear(self.embed_dim, self.embed_dim + 1, bias=False)
        self.lin2 = nn.Linear(self.embed_dim, self.embed_dim + 1, bias=False)
        self.lin3 = nn.Linear(self.embed_dim, self.embed_dim + 1, bias=False)
        self.lin4 = nn.Linear(self.embed_dim, self.embed_dim + 1, bias=False)
        # self.lin4 = nn.Linear(2 * (self.embed_dim+1), 1, bias=True)
        self.lin5 = nn.Linear(3 * (self.embed_dim+1), 1, bias=True)
        self.lin6 = nn.Linear(4 * (self.embed_dim+1), 1, bias=True)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        nn.init.xavier_uniform_(self.lin3.weight)
        nn.init.xavier_uniform_(self.lin4.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=lr_gamma)
        self.zero_pad = torch.zeros(1, self.embed_dim).cuda()
        self.cuda(device=0)

    def forward(self, features, adj_lists, seeds_idx_pad, seeds_idx_num, candidates_idx, all_nodes, mask, rest_idx, batch_size=1):
        embedding = self.sage(features, adj_lists, all_nodes, mask)
        embedding_pad = torch.cat((embedding, self.zero_pad), dim=0)
        row_index = torch.tensor([[i] for i in range(batch_size)])

        seeds_emb = embedding_pad[seeds_idx_pad]
        candidates_emb = embedding[candidates_idx]
        rest_emb = embedding[rest_idx]

        '''
        # seeds_mean = torch.mean(seeds_emb, dim=1).reshape(1, self.embed_dim+1)
        # global_mean = torch.mean(nn.Sigmoid()(self.lin2(embedding)), dim=0).reshape(1, self.embed_dim+1)   # embedding do not have batch
        # rest_mean = torch.mean(rest_emb, dim=1)

        seeds_mean =  nn.Sigmoid()(self.lin1(torch.mean(seeds_emb, dim=1)))
        global_mean = nn.Sigmoid()(self.lin2(torch.mean(embedding, dim=0).reshape(1, self.embed_dim)))
        rest_mean =  nn.Sigmoid()(self.lin3(torch.mean(rest_emb, dim=1)))
        candi_emb = nn.Sigmoid()(self.lin4(candidates_emb))

        if batch_size is not None:
            seeds_mean = torch.cat([seeds_mean for i in range(batch_size)], dim=0)
            global_mean = torch.cat([global_mean for i in range(batch_size)], dim=0)
        s_c_mean = nn.Sigmoid()(self.lin1(torch.mean(torch.cat((seeds_emb, candidates_emb.reshape(1,-1, self.embed_dim)), dim=1), dim=1)))

        # gain_frac = F.cosine_similarity(s_c_mean, global_mean ,dim=1)
        gain = self.lin5(torch.cat((seeds_mean, candi_emb, global_mean), dim=1))
        gain_overlap_frac =  F.cosine_similarity(seeds_mean, candi_emb, dim=1).reshape(batch_size, 1)
        # left_frac =  F.cosine_similarity(rest_mean, global_mean, dim=1)
        left_overlap_frac =  F.cosine_similarity(s_c_mean, rest_mean, dim=1).reshape(batch_size, 1)

        Q = torch.add(torch.mul(gain, 1-gain_overlap_frac) , torch.mul(log(self.graph_size)/gain, 1-left_overlap_frac) ) 
        '''





        '''

        # seeds_mean = nn.Sigmoid()(self.lin1(torch.sum(seeds_emb, dim=1).div(seeds_idx_num)))
        seeds_mean = nn.Sigmoid()(self.lin1(torch.mean(seeds_emb, dim=1)))
        # seeds_mean = torch.mean(seeds_emb, dim=1)
        global_mean = nn.Sigmoid()(self.lin2(torch.mean(embedding, dim=0).reshape(1, self.embed_dim)))
        # global_mean = torch.mean(embedding, dim=0).reshape(1, self.embed_dim)
        # candi_emb = candidates_emb
        candi_emb = nn.Sigmoid()(self.lin3(candidates_emb))

        if batch_size is not None:
            seeds_mean = torch.cat([seeds_mean for i in range(batch_size)], dim=0)
            global_mean = torch.cat([global_mean for i in range(batch_size)], dim=0)

        
        
        candidates_size = len(candidates_idx)
        cur_reward_frac = torch.sum(torch.mul(candi_emb, seeds_mean), dim=1)

        # Q = self.lin5(torch.cat((seeds_mean, global_mean, candi_emb), dim =1))
        # 某些时候有效
        Q = torch.sum(torch.mul(candi_emb, seeds_mean), dim=1)
        # Q = nn.(torch.sum(torch.mul(candi_emb, seeds_mean), dim=1))

        '''
        # seeds_mean = nn.ReLU(inplace=True)(self.lin1(torch.mean(seeds_emb, dim=1)))
        seeds_mean = nn.Sigmoid()(self.lin1(torch.mean(seeds_emb, dim=1)))
        # global_mean = nn.ReLU(inplace=True)(self.lin2(torch.mean(embedding, dim=0).reshape(1, self.embed_dim)))
        global_mean = nn.Sigmoid()(self.lin2(torch.mean(embedding, dim=0).reshape(1, self.embed_dim)))
        # rest_mean =  nn.ReLU(inplace=True)(self.lin3(torch.mean(rest_emb, dim=1)))
        rest_mean =  nn.Sigmoid()(self.lin3(torch.mean(rest_emb, dim=1)))
        # candi_emb = nn.ReLU(inplace=True)(self.lin4(candidates_emb)).reshape(batch_size, self.embed_dim+1)
        candi_emb = nn.Sigmoid()(self.lin4(candidates_emb))
        # seeds_mean = torch.mean(seeds_emb, dim=1)
        # global_mean = torch.mean(embedding, dim=0).reshape(1, self.embed_dim)
        # rest_mean = torch.mean(rest_emb, dim=1)
        # candi_emb = candidates_emb

        if batch_size is not None:
            seeds_mean = torch.cat([seeds_mean for i in range(batch_size)], dim=0)
            global_mean = torch.cat([global_mean for i in range(batch_size)], dim=0)

        
        margin_gain1 = torch.sub(candi_emb, seeds_mean)      
        margin_gain2 = torch.sub(rest_mean, seeds_mean)      

        Q = torch.sum(torch.mul(margin_gain1+margin_gain2, global_mean), dim=1)
        # Q = self.lin5(torch.cat((seeds_mean, global_mean, candi_emb), dim =1))
        # 某些时候有效
        # Q = self.lin4(torch.cat((margin_gain1, margin_gain2), dim=1))
        # Q = torch.sum(torch.mul(candi_emb, seeds_mean), dim=1)



  

        return Q










        

