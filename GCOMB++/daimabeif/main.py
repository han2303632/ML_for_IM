from Utils import Util
from Agent import Agent
from Env import Env
import numpy as np
from collections import defaultdict, namedtuple
import torch
from math import log, log10

numEpisode = 50
k = 20
windowSize = 2
emb_size = 2
hidden_size = 200
gcn_emb_size = 128
lr = 0.0005
gamma = 0.8
batch_size = 6
mem_size = 3 * k
graph_size = 776564     # 还没改啊
model_dir = "./model/model-%d.pkl"%(k)
# model_dir = "./model/model-%d-best2.pkl"%(k)
# model_dir = "/home/hunt/code4diff/model/model-%d.pkl"%(k)
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'input_dim',
     'output_dim' # the output (i.e., hidden) dimension
    ])
layer_infos = [SAGEInfo("node", None, 15, 3, 128),
        SAGEInfo("node", None, 20, 128, 128)]

def run():
    # 加载数据集
    util = Util()
    top_nodes_idx, adj_lists, edge_weight, features, node2idx, idx2node, num_used_nodes = util.load_data("./dataset/train/", k)
    num_top_nodes = len(top_nodes_idx)

    
    agent = Agent(model_dir, k, lr, gamma, mem_size, graph_size, layer_infos)
    env = Env(node2idx, idx2node, adj_lists, num_used_nodes)
    reward  = 0
    best_reward = 0


    for episode in range(numEpisode):
        print("episode:", episode)
        env.reset(top_nodes_idx)
        agent.reset(features, adj_lists, edge_weight, top_nodes_idx)
        for step in range(k):
            action,_ = agent.choose_action(step, env.seeds, env.candidates, env.mask)

            reward = env.react(episode, action, dataset="train")

            agent.reward_history.append(reward)

            if(step >= windowSize):
                if step == windowSize:
                    long_term_reward = log(agent.reward_history[step-1])
                else:
                    long_term_reward = agent.reward_history[step-1] - agent.reward_history[step-windowSize-1]
                # long_term_reward = log(long_term_reward + 1)
                long_term_reward = log10(long_term_reward+1)/log10(graph_size)
                seeds_prev, seeds_num_prev, action_prev, mask_prev, rest_idx = agent.state_history[step-windowSize];

                # if long_term_reward > 100:
                #     long_term_reward = 100
                

                # agent.memorize(episode, step, env.seeds.copy(), env.candidates.copy(), long_term_reward, mu_s.clone(), mu_c.clone(), mu_v.clone())
                # seeds_pair_cur = (env.seeds + [-1 for i in range(k - len(env.seeds))], len(env.seeds))
                seeds_cur = env.seeds.copy()
                candidates_cur = env.candidates.copy()
                # candidates_pair_cur = (env.candidates + [-1 for i in range(num_top_nodes - len(env.candidates))], len(env.candidates))
                # 注意reward 的位置
                agent.memorize(seeds_prev, seeds_num_prev, action_prev , mask_prev, rest_idx, step, seeds_cur, candidates_cur, long_term_reward, env.mask.clone())


                # if agent.memory.positive_counter > batch_size and agent.memory.negative_counter > batch_size:
            if agent.memory.counter >= batch_size:
                agent.learn(batch_size)

        if True:
            reward, _ = predict(agent)
            # reward, _ = validate(agent, top_nodes_idx, adj_lists, node2idx, idx2node, num_used_nodes)
            if reward > best_reward:
                torch.save(agent.Qfunc, model_dir)
                best_reward = reward
                # if episode % 2 == 0:
                #     agent.target_Q.load_state_dict(agent.Qfunc.state_dict())
                #     print("replace")

            # agent.Qfunc.scheduler.step()






def validate(agent, features, top_nodes_idx, adj_lists, edge_weight, node2idx, idx2node, num_used_nodes, dataset_type="test"):
    env = Env(node2idx, idx2node, adj_lists, num_used_nodes)
    reward = 0
    env.reset(top_nodes_idx)
    agent.reset(features, adj_lists, edge_weight, top_nodes_idx)

    for step in range(k):
        action, value = agent.choose_action(step, env.seeds, env.candidates, env.mask, task = "predict")
        if step == k-1:
            # reward = env.react(0, action, task="test", ignore_reward=False)
            reward = env.react(0, action, dataset=dataset_type, ignore_reward=False)
        else:
            # reward = env.react(0, action, task="test", ignore_reward=True)
            reward = env.react(0, action, dataset=dataset_type, ignore_reward=True)

    seeds =  [idx2node[node_idx] for node_idx in env.seeds]
    print("----------------------validation ----------------------")
    print("seeds", seeds)
    
    print("reward", reward )
    print("value", value)
    print("-------------------------------------------------------")
    return reward, seeds


def predict(agent=None):
    util = Util()
    g_size = 1079712

    top_nodes_idx, adj_lists, edge_weight, features, node2idx, idx2node, num_used_nodes = util.load_data("./dataset/test/", k) 
    # top_nodes_idx, adj_lists, nodes_nbr_weight, features, node2idx, idx2node, num_used_nodes = util.load_data("./dataset/train/", k) 
    if agent is None:
        agent = Agent(model_dir, k, lr, gamma, mem_size, g_size, layer_infos)
    reward, seeds = validate(agent, features, top_nodes_idx, adj_lists, edge_weight, node2idx, idx2node, num_used_nodes, dataset_type="test")
    # reward, seeds = validate(agent, features, top_nodes_idx, adj_lists, node2idx, idx2node, num_used_nodes, dataset_type="train")

    return reward,  seeds

            




if __name__ == "__main__":
    run()
    # predict()
