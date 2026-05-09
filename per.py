import numpy as np
import random

class SumTree:
   
    data_pointer = 0
    
    n_entries = 0 

    def __init__(self, capacity):
        self.capacity = capacity  
        
        self.tree = np.zeros(2 * capacity - 1)
        
        self.data = np.zeros(capacity, dtype=object)
    
        self.n_entries = 0 
        self.data_pointer = 0

    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_index, priority):
      
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, v):
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            if v <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                v -= self.tree[left_child_index]
                parent_index = right_child_index
                
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  


class PrioritizedReplayBuffer:
   
    e = 0.01  
    a = 0.6   
    beta = 0.4 
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    
    def __len__(self):
        return self.tree.n_entries

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        if self.tree.total_priority == 0:

             segment = 0
        else:
             segment = self.tree.total_priority / n
        
        priorities = []

        
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

      
        sampling_probabilities = np.array(priorities) / (self.tree.total_priority + 1e-5) # 0 나누기 방지
        is_weight = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        
      
        max_weight = is_weight.max()
        if max_weight == 0:
             is_weight = np.ones_like(is_weight) 
        else:
             is_weight /= max_weight 

        return batch, idxs, is_weight

    def update(self, idx, error):
      
        p = self._get_priority(error)
        self.tree.update(idx, p)