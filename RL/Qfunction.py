import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

class Qfunction(nn.Module):
    def __init__(self, embed_size,lr,  k, hidden_size, gcn_emb_size):
        super(Qfunction, self).__init__()
        self.embed_size = embed_size
        self.gcn_emb_size = gcn_emb_size

        # self.lin1 = nn.Linear(gcn_emb_size, embed_size+1, bias=False)
        self.lin1 = nn.Linear(embed_size, embed_size+1, bias=False)
        self.lin2 = nn.Linear(embed_size, embed_size+1, bias=False)
        self.lin3 = nn.Linear(embed_size, embed_size+1, bias=False)
        self.lin4 = nn.Linear(3 * (embed_size + 1) , 1, bias=True)
        # self.lin4 = nn.Linear(3 * (embed_size + 1) + hidden_size * 2, 1, bias=False)
        self.lin5 = nn.Linear(2 * hidden_size , hidden_size, bias=False)
        self.lin6 = nn.Linear(gcn_emb_size, hidden_size)

        self.lstm = torch.nn.LSTM(input_size=gcn_emb_size, hidden_size=hidden_size, batch_first=True) 
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.cuda(device=0)



    def forward(self, seeds_max_mat, candidates_max_mat, candidates_emb, seeds_gcn_mat, node_gcn_emb, batch_size):

        # origin
        Cat = torch.cat((self.lin1(candidates_max_mat), self.lin2(seeds_max_mat), self.lin3(candidates_emb)), dim = 1)
        Q = self.lin4(Cat).reshape(batch_size)
  
        ''' removed
        output = self.lstm(seeds_gcn_mat)[0][:, -1, :]
        # Cat1 = torch.nn.functional.softplus(torch.sum(torch.mul(output, self.lin6(node_gcn_emb)), 1))
        Cat1 = torch.cat((torch.sigmoid(self.lin6(node_gcn_emb)),  output), dim = 1)
        Cat = torch.cat((self.lin1(candidates_max_mat), self.lin2(seeds_max_mat), self.lin3(candidates_emb), Cat1), dim = 1)
        Q = self.lin4(Cat).reshape(batch_size)
        # Q = torch.mul((1-Cat1), self.lin4(Cat).reshape(batch_size))
        # Q = torch.mul((1-Cat1), self.lin4(candidates_emb).reshape(batch_size))
        '''




        return Q










        

