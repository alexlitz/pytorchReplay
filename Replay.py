import torch
from torch.utils.data.dataset import Dataset
import gym
from collections import namedtuple

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

class ReplayDataset(Dataset):
    """
    Dataset to for experience replay with pytorch and openai gym
    Purpose: easy experience replay.
    """

    def __init__(self, env, capacity):
        super(ReplayDataset, self).__init__()
        self.full = False
        self.len = 0
        self.capacity = capacity
        stateDims = [capacity] + list(env.observation_space.shape)
        self.states = torch.FloatTensor(*stateDims)
        self.actions = torch.FloatTensor(capacity, env.action_space.n)
        self.rewards = torch.FloatTensor(capacity)
        self.next_states = torch.FloatTensor(*stateDims)
        self.dones = torch.ByteTensor(capacity)

    def __len__(self):
        return self.len

    def __insert__(self, i, *args):
        if isinstance(args[0], Transition):
            t = args[0]
        else:
            t = Transition(args)

        self.states[i] = torch.Tensor([t.state])
        self.actions[i] = torch.Tensor([t.action])
        self.rewards[i] = float(t.reward)
        self.next_states[i] = torch.Tensor([t.next_state])
        self.dones[i] = int(t.done)

    def __add__(self, *args):
        if isinstance(args[0], Transition):
            t = args[0]
        else:
            t = Transition(args)
            
        if not self.full:
            self.insert(self.len, t)
            self.len += 1
            self.full = self.len==self.capacity
        else:
            self.insert(random.randint(0, self.len), t)


    def __getitem__(self, idx):
        return Transition(self.states[i], self.actions[i], self.rewards[i], 
                self.next_states[i], self.dones[i])

