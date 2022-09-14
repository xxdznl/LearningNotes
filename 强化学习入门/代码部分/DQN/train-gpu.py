import gym
import numpy as np
import random
import torch
import torch.nn as nn
import time

import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
display = Display(visible=0, size=(400, 300))
display.start()

from agent import Agent

EPSILON_DECAY = 200000
EPSILON_START = 1
EPSILON_END = 0.02
n_episode = 5000
n_step = 1000
TARGET_UPDATE_FREQUENCY = 200
LEARNING_RATE = 1e-3
GAMMA = 0.99
MEMORY_SIZE = 8192
BATCH_SIZE = 64
env = gym.make("CartPole-v0")
state = env.reset()

n_state = len(state)
n_action = env.action_space.n

REWARD_BUFFER = np.empty(shape=n_episode)

start = time.time()
agent = Agent(n_state=n_state,
                n_action=n_action,
                learning_rate=LEARNING_RATE,
                gamma=GAMMA,
                memory_size=MEMORY_SIZE,
                batch_size=BATCH_SIZE)
agent.eval_net.to("cuda:0")
agent.target_net.to("cuda:0")
#tensor = torch.rand(4,4)
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda:0')
# print(f"Device tensor is stored on: {tensor.device}")

for episode_i in range(n_episode):
    episode_reward = 0
    for step_i in range(n_step):
        epsilon = np.interp(episode_i * n_step + step_i,[0,EPSILON_DECAY],[EPSILON_START,EPSILON_END])
        random_sample = random.random()
        if(random_sample <= epsilon):
          action = env.action_space.sample()
        else:
          state2 = torch.as_tensor(state,dtype=torch.float32)
          state2 = state2.to("cuda:0")
          #print(f"Device tensor is stored on: {state2.device}")
          action = agent.eval_net.action(state2)
        state_next, reward, done, info = env.step(action)
        #if terminated or truncated:
        #    done = True
        agent.memory.add(state,action,reward,state_next,done) # TODO

        state = state_next
        episode_reward += reward

        if done:
          state = env.reset()
          REWARD_BUFFER[episode_i] = episode_reward
          break
        if np.mean(REWARD_BUFFER[:episode_i]) >= 120: 
          PATH = './DqnOnCartPole-v0.pth'
          torch.save(agent.target_net.state_dict(), PATH)
          exit()
          # count = 0
          # while True:
          #   state2 = torch.as_tensor(state,dtype=torch.float32)
          #   state2 = state2.to("cuda:0")
          #   a = agent.eval_net.action(state2)
          #   state, r, done, info = env.step(a)
          #   print(count,a)
          #   count += 1
          #   plt.imshow(env.render(mode='rgb_array'))# CHANGED
          #   ipythondisplay.clear_output(wait=True) # ADDED
          #   ipythondisplay.display(plt.gcf()) # ADDED

          #   if done:
          #     count = 0
          #     env.reset()

        batch_state,  batch_action, batch_reward, batch_state_next, batch_done = agent.memory.sample()
        
        batch_state = batch_state.to("cuda:0")
        batch_state_next = batch_state_next.to('cuda:0')
        batch_reward = batch_reward.to("cuda:0")
        batch_done = batch_done.to("cuda:0")
        batch_action = batch_action.to("cuda:0")
        
        target_Q = agent.target_net(batch_state_next) 
        max_target_Q = target_Q.max(dim = 1,keepdim = True)[0]
        y = batch_reward + agent.GAMMA * (1-batch_done) * max_target_Q 
        
        eval_Q = agent.eval_net(batch_state)  
        a_eval_Q = torch.gather(input = eval_Q,dim= 1,index=batch_action) #??????????????????????????

        loss = nn.functional.smooth_l1_loss(a_eval_Q,y)
        
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step() 

        #if step_i % TARGET_UPDATE_FREQUENCY == 0: # TODO 写循环里和外 看看区别
        #    agent.target_net.load_state_dict(agent.eval_net.state_dict()) 

    if (episode_i+1) % 10 == 0:
        # Print the training progress
        agent.target_net.load_state_dict(agent.eval_net.state_dict()) 
        #print("episode: {}".format(episode_i))
        print(f'episode:{episode_i+1}')
        print(f'time:',time.time()-start)
        print("Avg. Reward: {}".format(np.mean(REWARD_BUFFER[:episode_i])))
        print()
    if (episode_i+1) % 100 == 0:
      print(f'episode:{episode_i+1} time:',time.time()-start)
      print()

PATH = './DqnOnCartPole-v0.pth'
torch.save(agent.target_net.state_dict(), PATH)

#agent.target_net.load_state_dict(torch.load(PATH))
#agent.eval_net.load_state_dict(torch.load(PATH))