import torch
import numpy as np
import torch.nn as nn
import os
import random
import torch.utils.data as Data
from ReplayBuffer import ReplayBuffer
from Qfunction import Qfunction



class Agent:
    def __init__(self, gcn_embedding, embed_size, gamma, lr, mem_size, model_dir, k, hidden_size, gcn_emb_size):
        self.reward_history = []
        self.s_c_v_emb = []
        self.gamma = gamma
        self.embed_size = embed_size
        self.gcn_embedding = gcn_embedding
        self.gcn_embedding.cuda(device=0)
        self.k = k

        if os.path.exists(model_dir):
            print("load model:", model_dir)
            self.Qfunc = torch.load(model_dir)
        else:
            self.Qfunc = Qfunction(embed_size, lr, k, hidden_size, gcn_emb_size)
        # self.Qfunc.cuda(device=0)
        self.memory = ReplayBuffer(mem_size)



    def reset(self):
        self.reward_history = []
        self.s_c_v_emb = []

    # 做出某个行为
    def choose_action(self, step, embedding,  seeds_idx, candidates_idx, task="train"):
        action = 0
        reward = 0
        epsilon = max(0.5, 0.9**(step+1))   # 感觉这里有问题
        randOutput = np.random.rand()
        candidates_size = len(candidates_idx)

        seeds_max_mat, candidates_max_mat, candidates_emb = self.max_stack(embedding,  seeds_idx, candidates_idx) 

        seeds_gcn_emb = self.gcn_embedding_lookup(seeds_idx, step, self.k)
        seeds_gcn_mat = torch.stack([seeds_gcn_emb.clone() for i in range(candidates_size)], 0)
        candidates_gcn_mat = self.gcn_embedding[torch.tensor(candidates_idx, dtype=torch.long)]

        Q = self.Qfunc(seeds_max_mat.cuda(device=0), candidates_max_mat.cuda(device=0), candidates_emb.cuda(device=0), seeds_gcn_mat.cuda(device=0), candidates_gcn_mat.cuda(device=0), candidates_size)
        #  Q = self.Qfunc(seeds_max_mat, candidates_max_mat, candidates_emb, seeds_emb_mat, candidates_size)
        q_value, q_action = torch.sort(Q, descending=True)


        # finding the best node
        for act,value in zip(q_action,q_value):
            act = int(act)
            value = float(value)
            # if act not in seeds_idx:
            action = candidates_idx[act]
            reward = value
            break

        if task=="train" and (randOutput < epsilon or step == 0):
            print("rand choise")
            action = np.random.choice(candidates_idx, size=1)[0]

        # 预测阶段，第一个点为质量最大的点
        # if task=="predict" and step == 0:
        #     action = candidates_idx[0]

        # self.seeds_emb[step] = embedding[action].clone()
        # self.seeds_emb = torch.cat((self.seeds_emb[torch.randperm(step+1)],self.seeds_emb[step+1:]),dim=0)
        # save seed cand v embed
        if task == "train":
            self.s_c_v_emb.append([seeds_max_mat[0].clone(), candidates_max_mat[0].clone(), embedding[action].clone(), seeds_gcn_emb, self.gcn_embedding[action]])

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


    def gcn_embedding_lookup(self, seeds_idx, step, k):

        new_seeds_idx = seeds_idx.copy()
        random.shuffle(new_seeds_idx)
        final_idx =  new_seeds_idx + [-1 for i in range(k-step)]
        seeds_gcn_emb = self.gcn_embedding[torch.tensor(final_idx, dtype=torch.long)]

        return seeds_gcn_emb.view(k, -1)

        '''
        if task in ["train", "predict"]:
            # random.shuffle(seeds_idx)
            final_idx = seeds_idx
            # final_idx = seeds_idx + [-1 for i in range(k-step)]
        else:
            final_idx = seeds_idx

        seeds_gcn_emb_pad = self.gcn_embedding[torch.tensor([-1])]
        pad_idx = -1 * torch.ones((1,k-step), dtype=torch.long)
        if step == 0:
            seeds_gcn_emb_pad = gcn_embedding[pad_idx]
        else:
            seeds_gcn_emb = gcn_embedding[torch.tensor(seeds_idx)]
            padding = gcn_embedding[pad_idx]
            seeds_gcn_emb_pad = torch.cat((seeds_gcn_emb[torch.randperm(step)], padding), dim=0)
            # 随机打乱种子结点
        return final_idx, seeds_gcn_emb_pad.view(k, -1)
        '''


    # 记住一个历史元组
    def memorize(self, *args):
        self.memory.store(*args)


    # 调整参数
    def learn(self, embedding_history, batch_size):

        y_train = []
        mu_s_list = []
        mu_c_list = []
        mu_v_list = []
        gcn_s_list= []
        gcn_v_list = []
        # seeds_emb_list = []


        '''
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

        print("parameter update")
        for epoch in range(3):
            for episode, step, seeds_idx, candidates_idx, long_term_reward, mu_s, mu_c, mu_v in batch:
                _, pred_reward = self.choose_action(step, embedding_history[episode][step],  seeds_idx, candidates_idx, eval_loss=True)
                y_train = [long_term_reward + self.gamma*pred_reward]
                mu_s_list = [mu_s]
                mu_c_list = [mu_c]
                mu_v_list = [mu_v]
                self.Qfunc.optimizer.zero_grad()
                Q = self.Qfunc(torch.stack(mu_s_list, 0), torch.stack(mu_c_list, 0), torch.stack(mu_v_list, 0), 1)
                loss = torch.mean(torch.pow(Q - torch.tensor(y_train), 2))
                if epoch == 2:
                    print(Q)
                    print(y_train)
                    print("loss", loss)
                loss.backward()
                self.Qfunc.optimizer.step()
        '''


        batch = self.memory.sampling(batch_size)

        for episode, step, seeds_idx, candidates_idx, long_term_reward, mu_s, mu_c, mu_v, gcn_s, gcn_v in batch:
            _, pred_reward = self.choose_action(step, embedding_history[episode][step],  seeds_idx, candidates_idx, task="optimize")
            y_train.append(long_term_reward + self.gamma*pred_reward)
            mu_s_list.append(mu_s)
            mu_c_list.append(mu_c)
            mu_v_list.append(mu_v)
            gcn_s_list.append(gcn_s)
            gcn_v_list.append(gcn_v)

        torch_dataset = Data.TensorDataset(torch.stack(mu_s_list, 0).cuda(device=0), torch.stack(mu_c_list, 0).cuda(device=0), 
                                            torch.stack(mu_v_list, 0).cuda(device=0), torch.stack(gcn_s_list, 0).cuda(device=0), 
                                            torch.stack(gcn_v_list,0).cuda(device=0), torch.tensor(y_train).cuda(device=0))
        loader = Data.DataLoader(dataset=torch_dataset,
                                shuffle=False,
                                batch_size = 3)

        for epoch in range(100):
            for step, (s_batch, c_batch, v_batch, gcn_s_batch, gcn_v_batch, y_batch) in enumerate(loader):
                self.Qfunc.optimizer.zero_grad()
                Q = self.Qfunc(s_batch, c_batch, v_batch, gcn_s_batch, gcn_v_batch,3)
                loss = torch.mean(torch.pow((Q - y_batch), 2))
                loss.backward()
                self.Qfunc.optimizer.step()

                if epoch == 0:
                    print(loss)




        
