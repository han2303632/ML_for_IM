import numpy as np

class ReplayBuffer():
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.counter = 0
        # self.mus = [None] * mem_size
        # self.actions = [None] * mem_size
        # self.pre_states = [None] * mem_size
        # self.rewards = [None] * mem_size
        self.tuple = [None] * mem_size
    
    def store(self, episode, step, seeds_idx, candidates_idx, reward, mu_s, mu_c, mu_v):

        idx = self.counter % self.mem_size

        self.tuple[idx] = [episode, step, seeds_idx, candidates_idx
                            ,reward, mu_s, mu_c, mu_v]


        self.counter += 1

    def sampling(self, batch_size):
        max_mem = min(self.mem_size, self.counter)
        batch = np.random.choice(max_mem, batch_size, replace=True)
        return [self.tuple[i] for i in batch]

