from torch.utils.tensorboard import SummaryWriter # display the log

import gym # openai gym
import numpy as np 
import argparse

from torch.distributions import Normal
import torch.nn.functional as F # for mean square error
import torch.nn as nn # for neural network

from DDPG.Agent import Agent



class main():
    def __init__(self,args):
        env = gym.make(args.env_name)    # The wrapper encapsulates the gym env
        #num_states = env.observation_space.shape[0]
        #num_actions = env.action_space.shape[0]
        num_achive = env.observation_space['achieved_goal'].shape[0]
        num_desire = env.observation_space['desired_goal'].shape[0]
        num_obs =  env.observation_space['observation'].shape[0]
        num_actions = env.action_space.shape[0]
        num_states = num_obs + num_desire + num_achive
        print(num_actions)
        print(num_states)
        # args
        args.num_actions = num_actions
        args.num_states = num_states

        # create a enviroment
        env = Env(args.env_name)

        # print args 
        print("---------------")
        for arg in vars(args):
            print(arg,"=",getattr(args, arg))
        print("---------------")

        # create agent
        hidden_layer_num_list = [64,64]
        agent = Agent(args,env,hidden_layer_num_list)

        # trainning
        agent.train() 
        agent.state_norm.save_yaml(args.env_name+".yaml")

        # evaluate 
        env_evaluate = Env(args.env_name,render_mode='human')
        
        for i in range(10000):
            evaluate_reward = agent.evaluate_policy(env_evaluate)
            print("Evaluate reward:",evaluate_reward)


class Env(gym.Env):
    def __init__(self,env_name,render_mode=None):
        if render_mode == None:
            self.env = gym.make(env_name,reward_type="None")    # The wrapper encapsulates the gym env
        else:
            self.env = gym.make(env_name,render_mode='human',reward_type="None")    # The wrapper encapsulates the gym env
    def step(self, action):
        state, reward, done,truncated , info = self.env.step(action)   # calls the gym env methods
        ach = state['achieved_goal']
        des = state['desired_goal']
        obs = state['observation']
        state = np.hstack([obs,des,ach])
        return state, reward, done , truncated

    def reset(self):
        state = self.env.reset()[0]
        ach = state['achieved_goal']
        des = state['desired_goal']
        obs = state['observation']
        state = np.hstack([obs,des,ach])
        return state

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--var", type=float, default=3, help="Normal noise var")
    parser.add_argument("--tau", type=float, default=0.001, help="Parameter for soft update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--mem_min", type=float, default=64, help="minimum size of replay memory before updating actor-critic.")
    parser.add_argument("--env_name", type=str, default='FetchReach', help="Enviroment name")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=int(10000), help="Learning rate of actor")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Using state normalization.")
    parser.add_argument("--max_train_steps", type=int, default=int(3e4), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq_steps", type=float, default=2e3, help="Evaluate the policy every 'evaluate_freq' steps")
    args = parser.parse_args()

    
    main(args)