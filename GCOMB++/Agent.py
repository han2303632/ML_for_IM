import torch
import numpy as np
import torch.nn as nn
import os
import random
import torch.utils.data as Data
from ReplayBuffer import ReplayBuffer
from Qfunction import Qfunction



class Agent:
    def __init__(self, model_dir, features, adj_lists, k, lr, gamma, mem_size, top_nodes, num_top_nodes, graph_size, layer_infos, epsilon=1.0, eps_decay = 1E-4):
        self.reward_history = []
        self.state_history = []
        self.gamma = gamma
        self.k = k
        self.all_nodes = top_nodes
        self.num_top_nodes = num_top_nodes
        self.epsilon = epsilon
        self.eps_decay = eps_decay


        if os.path.exists(model_dir):
            print("load model:", model_dir)
            self.Qfunc = torch.load(model_dir)
        else:
            self.Qfunc = Qfunction(features, adj_lists, layer_infos, graph_size, lr)
        # self.Qfunc.cuda(device=0)
        self.memory = ReplayBuffer(mem_size)

    def reset(self):
        self.reward_history = []
        self.state_history = []

    # 做出某个行为
    def choose_action(self, step, seeds_idx, candidates_idx, all_nodes, mask, task="train"):
        action = 0
        reward = 0
        randOutput = np.random.rand()
        
        seeds_idx_pad = torch.tensor([seeds_idx + [-1 for i in range(self.k - len(seeds_idx))]]).cuda()
        seeds_idx_num = torch.tensor([[len(seeds_idx) if len(seeds_idx) > 0 else 1]]).cuda()
        Q = self.Qfunc(seeds_idx_pad, seeds_idx_num, torch.tensor(candidates_idx).cuda(), all_nodes, mask, batch_size=len(candidates_idx))
        q_value, q_action = torch.sort(Q, descending=True)


        # finding the best node
        for act,value in zip(q_action,q_value):
            act = int(act)
            # value = float(value)
            # if act not in seeds_idx:
            action = candidates_idx[act]
            reward = value
            break

        if task=="train" and (randOutput < self.epsilon or step == 0):
            print("rand choise")
            action = np.random.choice(candidates_idx, size=1)[0]
            self.epsilon = max(0.05, self.epsilon - self.eps_decay)

        # 预测阶段，第一个点为质量最大的点
        # if task=="predict" and step == 0:
        #     action = candidates_idx[0]

        # self.seeds_emb[step] = embedding[action].clone()
        # self.seeds_emb = torch.cat((self.seeds_emb[torch.randperm(step+1)],self.seeds_emb[step+1:]),dim=0)
        # save seed cand v embed
        if task == "train":
            self.state_history.append([
                seeds_idx + [-1 for i in range(self.k - len(seeds_idx))],
                len(seeds_idx) if len(seeds_idx) > 0 else 1,
                action,
                mask.clone()
            ])

        return action, reward;

            

    # 记住一个历史元组
    def memorize(self, *args):
        self.memory.store(*args)


    # 调整参数
    def learn(self, embedding_history, batch_size):

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
        

        for seeds_prev, seeds_num_prev, action_prev , mask_prev, step, seeds_cur, candidates_cur, long_term_reward, mask_cur in batch:
            Q_prev = self.Qfunc(torch.tensor([seeds_prev]).cuda(), torch.tensor([seeds_num_prev]).cuda(), torch.tensor([action_prev]).cuda(), self.all_nodes, mask_prev.cuda())
            _, pred_reard = self.choose_action(step, seeds_cur, candidates_cur, self.all_nodes, mask_cur, task="update")

            loss = torch.mean(torch.pow((Q_prev - (self.gamma * pred_reard + long_term_reward)), 2))
            print(loss)
            loss.backward()
            self.Qfunc.optimizer.step()
            

        '''    
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
        '''




        
