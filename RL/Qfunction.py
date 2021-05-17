import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

class Qfunction(nn.Module):
    def __init__(self, embed_size,lr):
        super(Qfunction, self).__init__()
        self.embed_size = embed_size

        self.lin1 = nn.Linear(embed_size, embed_size+1, bias=False)
        self.lin2 = nn.Linear(embed_size, embed_size+1, bias=False)
        self.lin3 = nn.Linear(embed_size, embed_size+1, bias=False)
        self.lin4 = nn.Linear(3 *(embed_size + 1), 1, bias=False)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)



    def forward(self, seeds_max_mat, candidates_max_mat, candidates_emb, batch_size):



        Cat = torch.cat((self.lin1(candidates_max_mat), self.lin2(seeds_max_mat), self.lin3(candidates_emb)), dim = 1)
        Q = self.lin4(Cat).reshape(batch_size)




        return Q










        

