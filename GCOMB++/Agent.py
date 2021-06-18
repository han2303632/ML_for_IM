import torch
import numpy as np
import torch.nn as nn
import os
import random
import torch.utils.data as Data
from ReplayBuffer import BiReplayBuffer,ReplayBuffer
from Qfunction import Qfunction



class Agent:
    def __init__(self, model_dir, k, lr, gamma, mem_size, num_top_nodes, graph_size, layer_infos, epsilon=0.8, eps_decay = 20E-4, replace_target=50):
        self.reward_history = []
        self.state_history = []
        self.gamma = gamma
        self.k = k
        self.num_top_nodes = num_top_nodes
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.target_Q = Qfunction(layer_infos, graph_size, lr)
        self.learn_step_cntr = 0
        self.replace_target = replace_target


        if os.path.exists(model_dir):
            print("load model:", model_dir)
            self.Qfunc = torch.load(model_dir)
        else:
            self.Qfunc = Qfunction(layer_infos, graph_size, lr)
        # self.Qfunc.cuda(device=0)
        self.memory = ReplayBuffer(mem_size)

    def reset(self):
        self.reward_history = []
        self.state_history = []

    # 做出某个行为
    def choose_action(self, step, features, adj_lists, seeds_idx, candidates_idx, all_nodes, mask, task="train"):
        action = 0
        reward = 0
        randOutput = np.random.rand()
        rest_idx_set = set(all_nodes) - set(seeds_idx)

        if seeds_idx == []:
            seeds_idx = [-1]
        
        # seeds_idx_pad = torch.tensor([seeds_idx + [-1 for i in range(self.k - len(seeds_idx))]]).cuda()
        seeds_idx_pad = torch.tensor([seeds_idx]).cuda()
        seeds_idx_num = torch.tensor([[len(seeds_idx) if len(seeds_idx) > 0 else 1]]).cuda()
        rest_idx = torch.tensor([list(rest_idx_set - set([idx])) for idx in candidates_idx]).cuda()
        # mask = torch.cat([mask for i in range(len(candidates_idx))], dim=0)

        # if task == "update":
        #     Q = self.target_Q(seeds_idx_pad, seeds_idx_num, torch.tensor(candidates_idx).cuda(), all_nodes, mask, rest_idx, batch_size=len(candidates_idx))
        # else:
        #     Q = self.Qfunc(seeds_idx_pad, seeds_idx_num, torch.tensor(candidates_idx).cuda(), all_nodes, mask, rest_idx, batch_size=len(candidates_idx))
        Q = self.Qfunc(features, adj_lists, seeds_idx_pad, seeds_idx_num, torch.tensor(candidates_idx).cuda(), all_nodes, mask, rest_idx, batch_size=len(candidates_idx))

        action, reward = self._max(Q, candidates_idx)

        if task=="train" and (randOutput < self.epsilon or step == 0):
            print("rand choise")
            action = np.random.choice(candidates_idx, size=1)[0]
            self.epsilon = max(0.05, self.epsilon - self.eps_decay)
            print("epsilon", self.epsilon)

        # 预测阶段，第一个点为质量最大的点
        # if task=="predict" and step == 0:
        #     action = candidates_idx[0]

        # self.seeds_emb[step] = embedding[action].clone()
        # self.seeds_emb = torch.cat((self.seeds_emb[torch.randperm(step+1)],self.seeds_emb[step+1:]),dim=0)
        # save seed cand v embed
        if task == "train":
            self.state_history.append([
                # seeds_idx + [-1 for i in range(self.k - len(seeds_idx))],
                seeds_idx.copy(),
                len(seeds_idx) if len(seeds_idx) > 0 else 1,
                action,
                mask.clone(),
                list(rest_idx_set - set([action]))
            ])

        return action, reward;

    def _max(self, Q, candidates_idx):
        action = None
        value = None

        q_value, q_action = torch.sort(Q, descending=True)

        for act,value in zip(q_action,q_value):
            act = int(act)
            # value = float(value)
            # if act not in seeds_idx:
            action = candidates_idx[act]
            reward = value
            break
        return action, reward


    # 记住一个历史元组
    def memorize(self, *args):
        self.memory.store(*args)


    # 调整参数
    def learn(self, features, adj_lists, all_nodes, batch_size):

        # self.learn_step_cntr += 1
        #if self.learn_step_cntr % self.replace_target == 1:
        #     print("replace./")
        #     self.target_Q.load_state_dict(self.Qfunc.state_dict())

        batch = self.memory.sampling(batch_size)
        for i in range(batch_size):
            seeds_prev, seeds_num_prev, action_prev , mask_prev, rest_idx, step, seeds_cur, candidates_cur, long_term_reward, mask_cur = batch[i]
            _, pred_reward = self.choose_action(step, features, adj_lists, seeds_cur, candidates_cur, all_nodes, mask_cur, task="update")
            batch[i] = [seeds_prev, seeds_num_prev, action_prev , mask_prev, rest_idx, step, seeds_cur, candidates_cur, long_term_reward, mask_cur, float(pred_reward)]

        for i in range(2):
            random.shuffle(batch)

            for seeds_prev, seeds_num_prev, action_prev , mask_prev, rest_idx, step, seeds_cur, candidates_cur, long_term_reward, mask_cur, pred in batch:
                self.Qfunc.optimizer.zero_grad()
                Q_prev = self.Qfunc(features, adj_lists, torch.tensor([seeds_prev]).cuda(), torch.tensor([seeds_num_prev]).cuda(), torch.tensor([action_prev]).cuda(), all_nodes, mask_prev.cuda(), torch.tensor([rest_idx]).cuda())

                # loss = torch.mean(torch.pow((Q_prev - (self.gamma * pred_reward + long_term_reward)), 2))
                loss = torch.mean(torch.pow(((self.gamma * pred + long_term_reward) - Q_prev), 2))
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

        '''




        
