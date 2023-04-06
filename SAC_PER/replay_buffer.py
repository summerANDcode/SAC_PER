import numpy as np
import torch


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, alpha=0.6,beta_start = 0.4,beta_frames=100000):
        self.capacity = capacity
        self.device = device
        # 控制优先级的两个参数
        self.alpha = alpha
        self.beta_start = beta_start

        self.beta_frames = beta_frames
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        # 类比于原始的self.buffer     = []
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        # 添加优先级,以及W
        self.prior = np.zeros((capacity, 1), dtype=np.float32)
        self.frame = 1 #for beta calculation
        self.pos        = 0

        self.idx = 0
        self.last_save = 0
        self.full = False

    # def __init__(self, capacity, alpha=0.6,beta_start = 0.4,beta_frames=100000):
    #
    #     self.buffer     = []
    #     self.pos        = 0
    #     self.priorities = np.zeros((capacity,), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.idx
    def updata(self):
        pass

    def add(self, obs, action, reward, next_obs, done, done_no_max):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

        max_p = np.max(self.prior[-self.capacity:])
        if max_p == 0:
            max_p = 1
        np.copyto(self.prior[self.idx], max_p)
        

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def UpdataPrior(self,p,indx):
        # for i in indx:
        #     self.prior[i] = p[i]
       for idx, prio in zip(indx, p):
            self.prior[idx] = abs(prio)

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.

        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent
        that reaches 1 only at the end of learning. In practice, we linearly anneal from its initial value 0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    def sample(self, batch_size):
        N = self.idx
        if N == self.capacity:
            prios = self.prior
        else:
            prios = self.prior[:self.idx]
        if N==10:
            print("self.prior",self.prior)
        # calc P = p^a/sum(p^a)
        probs = prios ** self.alpha
        P = probs / probs.sum()
        P=np.array(P).reshape(-1)
        
        # # gets the indices depending on the probability p
        idxs = np.random.choice(N, batch_size, p=P,replace=False)
        
        # samples = [self.buffer[idx] for idx in indices]
        #
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # idxs = np.random.randint(0,
        #                          self.capacity if self.full else self.idx,
        #                          size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        prior =  torch.as_tensor(self.prior[idxs], device=self.device)

        # Compute importance-sampling weight
        weights = (N * P[idxs]) ** (-beta)
        # normalize weightsss
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        # weights = torch.as_tensor(weights,device=self.device).float()
        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max, weights, idxs

    def printobs(self):
        for idx in range(self.idx):
            print('obs: ', self.obses[idx])

