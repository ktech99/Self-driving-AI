# AI for Self Driving Car

# Importing the libraries

import numpy as np #for arrays
import random
import os #for saving and loading file
import torch # using pytorch
import torch.nn as nn #pytorch neural network
import torch.nn.functional as F #creating a shortcut to functional
import torch.optim as optim # optimiser for gradient descent
import torch.autograd as autograd #convert tensor to gradient
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module): # inheriting from nn.module

    def __init__(self, input_size, nb_action): # defining architecture of nn
        #self is the object of this method
        # input_size = number of values for input(input neurond)
        #nb_action = how many actions can be taken
        super(Network, self).__init__()
        self.input_size = input_size #attaching to object
        self.nb_action = nb_action #attaching to object
        self.fc1 = nn.Linear(input_size, 40) #full connection input>hidden
        self.fc2 = nn.Linear(40, nb_action) #full connection hidden>action

    def forward(self, state):
        x = F.relu(self.fc1(state)) #hidden neurons
        #relu is rectifier function
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] #creating memory

    def push(self, event):
        self.memory.append(event) #adding event to memory
        if len(self.memory) > self.capacity:
            del self.memory[0] #deleting the first element

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
                #taking random samples from memory of bitch size
                #zip* reshapes list (1,2))(3,4) > (1,3)(2,4)
                #returning map function converting it to pytorch variable
                #lambda is function name
                #x is variable
                #: is what it returns
                #variable converts torch to tensor
                #torch.cat is concatinating x values
                # first dimension is 0
                # running for all samples
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # optimizer of adam class
        # lr is learning rate
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        #using tensor class
        #unsqueeze is fake dimension, 0 is first dimension
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state): #probabilities of Q values
        probs = F.softmax(self.model(Variable(state, volatile = True))*70) # T=100
        # as state is a torch tensor value
        # volatile = true, doesn't graph
        # temperature = 100, higher is the temp, higher is the prob of winning Q value
        action = probs.multinomial() # random draw based on probability
        return action.data[0,0] # saved in 0,0

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #gather 1 is gathering the action which is most probable
        #squeeze and unsqueeze are for fake dimensions
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        #maximum of Q values
        #action represented by index 1
        #next state represented by index
        target = self.gamma*next_outputs + batch_reward
        #target = gamma*nextOutput + batchReward
        td_loss = F.smooth_l1_loss(outputs, target)
        #temporal difference loss using hoover algorithm
        self.optimizer.zero_grad()
        #using adamoptimizer and reinitialising at each iteration
        td_loss.backward(retain_variables = True)
        #back propogation
        #retain variables is to save memory
        self.optimizer.step()
        #updates weights using optimizer

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]))) #updating memory
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100: #time to learn
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100) #learning from 100 samples
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action #updating last action
        self.last_state = new_state #updating last state
        self.last_reward = reward #updating reward
        self.reward_window.append(reward) #adding reward to reward window
        if len(self.reward_window) > 1000:
            del self.reward_window[0] #removing first element
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
        # totalsum/legth of reward window
        # adding 1 so denominator isn't 0

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
                   #saving last version of weight(model) and optimizer in dictionary in file named last_brain.pth

    def load(self):
        #checking if file exists
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])#loading saved values
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
