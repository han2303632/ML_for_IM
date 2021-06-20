import numpy as np
import random

class ReplayBuffer():
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.counter = 0
        # self.mus = [None] * mem_size
        # self.actions = [None] * mem_size
        # self.pre_states = [None] * mem_size
        # self.rewards = [None] * mem_size
        self.tuple = [None] * mem_size
    
    def store(self, *args):

        idx = self.counter % self.mem_size
        self.tuple[idx] = [*args]
        self.counter += 1

    def sampling(self, batch_size):
        max_mem = min(self.mem_size, self.counter)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return [self.tuple[i] for i in batch]

class BiReplayBuffer():
    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.positive_counter = 0
        self.negative_counter = 0
        self.positive_tuple = [None] * mem_size
        self.negative_tuple = [None] * mem_size
    
    def store(self, *args):
        if args[-2] != 0:
            idx = self.positive_counter % self.mem_size
            self.positive_tuple[idx] = [*args]
            self.positive_counter += 1
        else:
            idx = self.negative_counter % self.mem_size
            self.negative_tuple[idx] = [*args]
            self.negative_counter += 1
    def sampling(self, batch_size):
        positive_max_mem = min(self.mem_size, self.positive_counter)
        negative_max_mem = min(self.mem_size, self.negative_counter)
        positive_batch = np.random.choice(positive_max_mem, batch_size, replace=False)
        negative_batch = np.random.choice(negative_max_mem, batch_size, replace=False)

        batch = [self.positive_tuple[i] for i in positive_batch] + [self.negative_tuple[i] for i in negative_batch]
        random.shuffle(batch)

        return batch


