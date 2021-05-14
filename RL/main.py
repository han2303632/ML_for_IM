from Utils import Util
from Agent import Agent
from Env import Env
import numpy as np
import torch

numEpisode = 50
k = 30
# windowSize = 2
windowSize = 1
emb_size = 2
hidden_size = 30
learningRate = 0.0005
gamma = 0.8
batch_size = 6
model_dir = "./model/model-%d.pkl".format(k)


def run():
    # 加载数据集
    util = Util()
    graph, embedding, graph_size, top_nodes_idx, node2neighbors, idx2node, node2idx = util.load_dataset("./dataset/train/", k, 0.003)

    
    agent = Agent(emb_size, gamma, learningRate, k*3, model_dir)
    env = Env()
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
                    long_term_reward = agent.reward_history[step]
                else:
                    long_term_reward = agent.reward_history[step] - agent.reward_history[step-windowSize-1]
                mu_s, mu_c, mu_v = agent.s_c_v_emb[step-windowSize];

                agent.memorize(episode, step, env.seeds.copy(), env.candidates.copy(), long_term_reward, mu_s.clone(), mu_c.clone(), mu_v.clone())


                if agent.memory.counter > 5:
                    agent.learn(env.embedding_history, batch_size)

        if episode % 4 == 0 and episode != 0:
            reward, _ = validate(agent, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node)
            if reward > best_reward:
                torch.save(agent.Qfunc, model_dir)






def validate(agent, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node):
    env = Env()
    reward = 0
    env.reset(0, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node)
    agent.reset()

    for step in range(k):
        action, _ = agent.choose_action(step, env.embedding, env.seeds, env.candidates, eval_loss = True)
        reward = env.react(0, action)

    seeds =  [idx2node[node_idx] for node_idx in env.seeds]
    print("----------------------validation ----------------------")
    print("seeds", seeds)
    
    print("reward", reward )
    print("-------------------------------------------------------")
    return reward, seeds


def predict():
    util = Util()
    graph, embedding, graph_size, top_nodes_idx, node2neighbors, idx2node, node2idx = util.load_dataset("./dataset/", k, 0.003)

    
    agent = Agent(emb_size, gamma, learningRate, k*3, model_dir, hidden_size, k, batch_size)
    env = Env()
    reward, seeds = validate(agent, embedding, emb_size, graph, graph_size, top_nodes_idx, node2neighbors, node2idx, idx2node)

            




if __name__ == "__main__":
    run()
    # predict()