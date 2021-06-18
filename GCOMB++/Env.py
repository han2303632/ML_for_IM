import torch
import torch.nn as nn
import os
import copy
from sklearn.preprocessing import StandardScaler
class Env:

    def __init__(self, node2idx, idx2node, adj_lists, used_nodes_size):
        self.embedding_history = []
        self.node2idx = node2idx
        self.idx2node = idx2node
        self.adj_lists = adj_lists
        self.num_used_nodes = used_nodes_size

    def reset(self, candidates):
        self.candidates = candidates.copy()
        self.seeds = []
        self.mask = torch.BoolTensor([[1] for i in range(self.num_used_nodes)]).cuda()



    def react(self, episode, choosed_idx, task="train", ignore_reward=False):
        self.candidates.remove(choosed_idx)
        self.seeds.append(choosed_idx)
        self.mask[choosed_idx] = torch.tensor([0])
        for nbr_idx in self.adj_lists[choosed_idx]:
            self.mask[nbr_idx] = torch.tensor([0])
        
        '''
        # update locality
        node = self.idx2node[choosed_idx]
        node_nbr = self.node2neighbors[node]
        for key in self.node2neighbors.keys():
            if key != node:
                self.node2neighbors[key] = set(self.node2neighbors[key]) - set(node_nbr)
                self.embedding[self.node2idx[key]][0] = len(self.node2neighbors[key])
            else:
                self.node2neighbors[key] = set()
                self.embedding[choosed_idx][0] = 0



        # normalize embedding
        scaler = StandardScaler()
        scaler.fit(self.embedding)
        self.embedding = torch.from_numpy(scaler.transform(self.embedding)).float()

        # normalize after parameter update
        # self.embedding = torch.nn.functional.normalize(self.embedding, dim=0)
        self.embedding_history[episode].append(self.embedding.clone())
        '''

        reward = 0
        # calculate and save reward
        if not ignore_reward:
            reward = self.eval_single_step_reward(task)

        return reward


    def eval_single_step_reward(self, task):
        graph_dir = "../influence_evaluate/data/youtube_" + task + "/mc_sim"
        reward = -1
        command = "../influence_evaluate/influenceEval "  + "-dataset %s -seeds %s -task seed_eval"%(
            graph_dir, ",".join([str(self.idx2node[idx]) for idx in self.seeds])
            )
        print(command)

        for line in os.popen(command):
            if line.startswith("reward:"):
                reward = float(line.split(":")[1])
                break

        print(reward)

        assert reward != -1


        return reward

        




