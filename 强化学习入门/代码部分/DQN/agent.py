import numpy as np
import random
import torch
import torch.nn as nn

class ReplayMemory:
    def __init__(self,n_state,n_action,memory_size,batch_size) -> None:
        self.n_state = n_state
        self.n_action = n_action

        self.MEMORY_SIZE = memory_size
        self.BATCH_SIZE = batch_size

        self.all_state = np.empty(shape = (self.MEMORY_SIZE,self.n_state),dtype=np.float64) 
        self.all_action = np.random.randint(low=0,high=self.n_action,size=self.MEMORY_SIZE,dtype=np.uint8)
        self.all_reward = np.empty(self.MEMORY_SIZE,dtype=np.float64)
        self.all_state_next = np.empty(shape = (self.MEMORY_SIZE,self.n_state),dtype=np.float64)
        self.all_done = np.random.randint(low=0,high=2,size=self.MEMORY_SIZE,dtype=np.uint8) 
        self.max = 0
        self.t = 0

    def add(self,state,action,reward,state_next,done):
        self.all_state[self.t] = state
        self.all_action[self.t] = action
        self.all_reward[self.t] = reward
        self.all_state_next[self.t] = state_next
        self.all_done[self.t] = done
        self.max = max(self.max,self.t+1)
        self.t = (self.t + 1) % self.MEMORY_SIZE
    
    def sample(self):
        if self.max >= self.BATCH_SIZE:
            indexes = random.sample(range(0,self.max),self.BATCH_SIZE)
        else:
            indexes = range(0,self.max)
        batch_state = []
        batch_action = []
        batch_reward = []
        batch_state_next = []
        batch_done = []

        for idx in indexes:
            batch_state.append(self.all_state[idx])
            batch_action.append(self.all_action[idx])
            batch_reward.append(self.all_reward[idx])
            batch_state_next.append(self.all_state_next[idx])
            batch_done.append(self.all_done[idx])
        
        batch_state_tensor = torch.as_tensor(np.asarray(batch_state),dtype = torch.float32)
        batch_action_tensor = torch.as_tensor(np.asarray(batch_action),dtype = torch.int64).unsqueeze(-1)# 要与eval_Q维数匹配   torch.gather(input = eval_Q,dim= 1,index=batch_action)
        batch_reward_tensor = torch.as_tensor(np.asarray(batch_reward),dtype = torch.float32).unsqueeze(-1)
        batch_state_next_tensor = torch.as_tensor(np.asarray(batch_state_next),dtype = torch.float32)
        batch_done_tensor = torch.as_tensor(np.asarray(batch_done),dtype = torch.float32).unsqueeze(-1)
        return batch_state_tensor, batch_action_tensor, batch_reward_tensor, batch_state_next_tensor, batch_done_tensor

class DQN(nn.Module):
    def __init__(self,n_state,n_action) -> None:
        super().__init__()
        in_feature = n_state
        self.net = nn.Sequential(
            nn.Linear(in_feature,64),
            nn.Tanh(),
            nn.Linear(64,n_action)
        )
    def forward(self,x):
        return self.net(x)
    
    def action(self,state):
        state_tensor = torch.as_tensor(state,dtype=torch.float32) #这里的state是np.asarray类型不用再np.asarray转化
        Q_value  = self(state_tensor.unsqueeze(0))#代码中的state其实就是Q_value  不理解为什么加self()              ??????????????????????????????????????????????????
        #Q_value = state_tensor.unsqueeze(0)#不理解为什么要升维？？？？？？？？？
        max_Q_idx = torch.argmax(Q_value)
        action = max_Q_idx.detach().item()#不理解为什么要加detach()
        #action = max_Q_idx.item()
        return action
class Agent:
    def __init__(self,n_state,n_action,learning_rate,gamma,memory_size,batch_size) -> None:
        self.GAMMA = gamma
        self.learning_rate = learning_rate
        self.n_state = n_state
        self.n_action = n_action
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = ReplayMemory(n_state = self.n_state,n_action = self.n_action,memory_size= self.memory_size,batch_size= self.batch_size)

        self.eval_net = DQN(self.n_state,self.n_action)
        self.target_net = DQN(self.n_state,self.n_action)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=self.learning_rate)
