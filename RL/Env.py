import torch
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler
class Env:

    def __init__(self):
        self.embedding_history = []

    def reset(self, episode, embedding, embed_size, graph, g_size, candidates, node2neighbors, node2idx, idx2node):
        self.graph = graph
        self.g_size = g_size
        self.candidates = candidates.copy()
        self.seeds = []
        self.node2neighbors = node2neighbors.copy()
        self.node2idx = node2idx.copy()
        self.idx2node = idx2node.copy()

        self.embedding = torch.from_numpy(embedding).clone().float()
        self.embedding_history.append([])
        self.embedding_history[episode].append(self.embedding.clone());



    def react(self, episode, choosed_idx):
        self.candidates.remove(choosed_idx)
        self.seeds.append(choosed_idx)

        # update locality
        node = self.idx2node[choosed_idx]
        node_nbr = self.node2neighbors[node]
        for key in self.node2neighbors.keys():
            if key != node:
                self.node2neighbors[key] = self.node2neighbors[key] - node_nbr
                self.embedding[self.node2idx[key]][0] = len(self.node2neighbors[key])



        # normalize embedding
        # scaler = StandardScaler()
        # scaler.fit(self.embedding)
        # self.embedding = scaler.transform(self.embedding)

        # normalize after parameter update
        self.embedding = torch.nn.functional.normalize(self.embedding, dim=0)
        self.embedding_history[episode].append(self.embedding.clone())


        # calculate and save reward
        reward = self.eval_single_step_reward()

        return reward


    def eval_single_step_reward(self):
        graph_dir = "../influence_evaluate/data/youtube_train/mc_sim"
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

        




