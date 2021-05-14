import torch
import numpy as np
import torch.nn as nn
import os
from ReplayBuffer import ReplayBuffer
from Qfunction import Qfunction



class Agent:
    def __init__(self, embed_size, gamma, lr, mem_size, model_dir):
        self.reward_history = []
        self.s_c_v_emb = []
        self.gamma = gamma
        self.embed_size = embed_size

        if os.path.exists(model_dir):
            self.Qfunc = torch.load(model_dir)
        else:
            self.Qfunc = Qfunction(embed_size, lr)
        # self.Qfunc.cuda(device=0)
        self.memory = ReplayBuffer(mem_size)



    def reset(self):
        self.reward_history = []
        self.s_c_v_emb = []

    # 做出某个行为
    def choose_action(self, step, embedding,  seeds_idx, candidates_idx, eval_loss=False):
        action = 0
        reward = 0
        epsilon = max(0.05, 0.9**(step+1))   # 感觉这里有问题
        randOutput = np.random.rand()
        candidates_size = len(candidates_idx)

        seeds_max_mat, candidates_max_mat, candidates_emb = self.max_stack(embedding,  seeds_idx, candidates_idx) 

        Q = self.Qfunc(seeds_max_mat, candidates_max_mat, candidates_emb, candidates_size)
        #  Q = self.Qfunc(seeds_max_mat, candidates_max_mat, candidates_emb, seeds_emb_mat, candidates_size)
        q_value, q_action = torch.sort(Q, descending=True)


        # finding the best node
        for act,value in zip(q_action,q_value):
            act = int(act)
            value = float(value)
            if act not in seeds_idx:
                action = candidates_idx[act]
                reward = value
                break

        if not eval_loss and (randOutput < epsilon or step == 0):
            print("rand choise")
            action = np.random.choice(candidates_idx, size=1)[0]

        # 预测阶段，第一个点为质量最大的点
        # if eval_loss and step == 0:
        #     action = candidates_idx[0]

        # self.seeds_emb[step] = embedding[action].clone()
        # self.seeds_emb = torch.cat((self.seeds_emb[torch.randperm(step+1)],self.seeds_emb[step+1:]),dim=0)
        # save seed cand v embed
        if not eval_loss:
            self.s_c_v_emb.append([seeds_max_mat[0], candidates_max_mat[0], embedding[action].clone()])

        return action, reward;

            
    def max_stack(self, embedding,  seeds_idx, candidates_idx):
        candidates_size = len(candidates_idx)

        if len(seeds_idx) == 0:
            seeds_emb = -1000000 * torch.tensor([[1,1]], dtype=torch.float)
        else:
            seeds_emb = torch.index_select(embedding, 0, torch.tensor(seeds_idx, dtype=torch.long), out=None)
        candidates_emb = torch.index_select(embedding, 0, torch.tensor(candidates_idx, dtype=torch.long), out=None)
        
        # max
        seeds_max = torch.max(seeds_emb, 0)[0]
        candidates_max = torch.max(candidates_emb, 0)[0]

        # stack max
        seeds_max_list = torch.stack([seeds_max.clone() for i in range(candidates_size)], 0)
        candidates_max_list = torch.stack([candidates_max.clone() for i in range(candidates_size)], 0)



        return seeds_max_list, candidates_max_list, candidates_emb

    # 记住一个历史元组
    def memorize(self, episode, step, seeds_idx, candidates_idx, long_term_reward, mu_s, mu_c, mu_v):
        self.memory.store(episode, step, seeds_idx, candidates_idx, long_term_reward, mu_s, mu_c, mu_v)


    # 调整参数
    def learn(self, embedding_history, batch_size):

        y_train = []
        mu_s_list = []
        mu_c_list = []
        mu_v_list = []
        # seeds_emb_list = []

        batch = self.memory.sampling(batch_size)

        for episode, step, seeds_idx, candidates_idx, long_term_reward, mu_s, mu_c, mu_v in batch:
            _, pred_reward = self.choose_action(step, embedding_history[episode][step],  seeds_idx, candidates_idx, eval_loss=True)
            y_train.append(long_term_reward + self.gamma*pred_reward)
            mu_s_list.append(mu_s)
            mu_c_list.append(mu_c)
            mu_v_list.append(mu_v)


        self.Qfunc.optimizer.zero_grad()
        # Q = self.Qfunc(torch.stack(mu_s_list, 0).cuda(device=0), torch.stack(mu_c_list, 0).cuda(device=0), torch.stack(mu_v_list, 0).cuda(device=0), torch.stack(seeds_emb_list, 0).cuda(device=0), self.batch_size)
        Q = self.Qfunc(torch.stack(mu_s_list, 0), torch.stack(mu_c_list, 0), torch.stack(mu_v_list, 0), batch_size)

        loss = torch.mean(torch.pow(torch.tensor(y_train) - Q, 2))
        # loss = torch.mean(torch.pow(torch.tensor(y_train).cuda(device=0) - Q, 2))
        print("loss", loss)


        print("parameter update")
        loss.backward()
        self.Qfunc.optimizer.step()





        
