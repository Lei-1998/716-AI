import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

env = gym.make('CartPole-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9             # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 2000
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
print(N_ACTIONS, N_STATES)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.eval_net = self.eval_net.cuda()
        self.target_net = self.target_net.cuda()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = x.cuda()
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            # print(actions_value)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            # print(action)
            action = action[0]   # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).cuda()
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate

        # q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)

        max_action_next = self.eval_net(b_s_).max(1)[1].view(BATCH_SIZE, 1)
        #print(max_action_next)
        q_target_batch = []
        for i in range(BATCH_SIZE):
            #print(max_action_next[i])
            target_Q_value = q_next[i, max_action_next[i]]
            q_target_batch.append(b_r[i] + GAMMA * target_Q_value)
        q_target = torch.tensor([q_target_batch]).view(32, 1)

        q_eval = q_eval.cuda()
        q_target = q_target.cuda()
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()
for i_episiode in range(400):
    observation = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = dqn.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        x, x_vel, theta, theta_vel = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2
        dqn.store_transition(observation, action, reward, observation_)
        total_reward += reward
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep:', i_episiode, '|', 'total_reward', total_reward)

        if done:
            break

        observation = observation_

