from Utils import Util
from Agent import Agent
from Env import Env
import numpy as np
import torch

numEpisode = 50
k = 20
# windowSize = 2
windowSize = 2
emb_size = 2
hidden_size = 200
gcn_emb_size = 120
learningRate = 0.0005
gamma = 0.8
batch_size = 6
model_dir = "./model/model-%d.pkl"%(k)
# model_dir = "./model/model-%d-best.pkl"%(k)
# model_dir = "/home/hunt/code4diff/model/model-%d.pkl"%(k)


def run():
    # 加载数据集
    util = Util()
    graph, embedding, graph_size, top_nodes_idx, node2neighbors, idx2node, node2idx, gcn_embedding = util.load_dataset("./dataset/train/", k, 0.003)

    
    agent = Agent(gcn_embedding, emb_size, gamma, learningRate, k*3, model_dir, k, hidden_size, gcn_emb_size)
    env = Env(gcn_embedding)
    reward  =0
    best_reward = 0


    for episode in range(numEpisode):
        print("episode:", episode)
        env.reset(episode, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node)
        agent.reset()
        for step in range(k):
            action,_ = agent.choose_action(step, env.embedding, env.seeds, env.candidates)

            reward = env.react(episode, action)

            agent.reward_history.append(reward)

            if(step >= windowSize):
                if step == windowSize:
                    long_term_reward = agent.reward_history[step-1]/1000
                else:
                    long_term_reward = agent.reward_history[step-1] - agent.reward_history[step-windowSize-1]
                mu_s, mu_c, mu_v, gcn_s , gcn_v= agent.s_c_v_emb[step-windowSize];
                

                # agent.memorize(episode, step, env.seeds.copy(), env.candidates.copy(), long_term_reward, mu_s.clone(), mu_c.clone(), mu_v.clone())
                agent.memorize(episode, step, (env.seeds[:-1]).copy(), (env.candidates + [action]).copy(), long_term_reward, mu_s, mu_c, mu_v, gcn_s, gcn_v)


                if agent.memory.counter > 5:
                    agent.learn(env.embedding_history, batch_size)

        if episode % 2 == 0 and episode != 0:
            reward, _ = validate(agent, gcn_embedding, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node)
            if reward > best_reward:
                torch.save(agent.Qfunc, model_dir)
                best_reward = reward






def validate(agent, gcn_embedding, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node):
    env = Env(gcn_embedding)
    reward = 0
    env.reset(0, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node)
    agent.reset()

    for step in range(k):
        action, _ = agent.choose_action(step, env.embedding, env.seeds, env.candidates, task="predict")
        reward = env.react(0, action)

    seeds =  [idx2node[node_idx] for node_idx in env.seeds]
    print("----------------------validation ----------------------")
    print("seeds", seeds)
    
    print("reward", reward )
    print("-------------------------------------------------------")
    return reward, seeds


def predict():
    util = Util()
    graph, embedding, graph_size, top_nodes_idx, node2neighbors, idx2node, node2idx, gcn_embedding = util.load_dataset("./dataset/train/", k, 0.003)

    
    agent = Agent(gcn_embedding, emb_size, gamma, learningRate, k*3, model_dir, k, hidden_size, gcn_emb_size)
    env = Env(gcn_embedding)
    reward, seeds = validate(agent, gcn_embedding, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node)

            




if __name__ == "__main__":
    run()
    # predict()
