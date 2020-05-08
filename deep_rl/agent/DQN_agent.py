#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *

from scipy.linalg import eigh,svd
import matplotlib.pyplot as plt

class DQNActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.v = 0
        self.linear_weights = 0

    def _transition(self):
        '''if self._state is None:
            self._state = self._task.reset()

        config = self.config
        with config.lock:
            q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry '''

        if self._state is None:
            self._state = self._task.reset()

        #state_indices = np.argmax(self._state)
        #states = self.v[:,state_indices,:]
        #q = (torch.matmul(torch.from_numpy(states).float(),self.linear_weights)).t()
        #print(q)

        config = self.config

        if config.control == True:
            if self._total_steps < config.exploration_steps \
                    or np.random.rand() < config.random_action_prob():
                action = np.random.randint(0, 4)
            else:
                state_indices = np.argmax(self._state)
                states = self.v[:,state_indices,:]
                q_values = (torch.matmul(torch.from_numpy(states).float(),self.linear_weights)).t()
                action = torch.argmax(q_values)
            next_state, reward, done, info = self._task.step([action])
            entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
            self._total_steps += 1
            self._state = next_state
            return entry

        elif config.control == False:
            state_indices = np.argmax(self._state)
            if np.random.rand() < config.random_action_prob():
                action = np.random.randint(0, 4)
            else:
                action = config.fixed_policy[config.obs_map == state_indices]
            action = int(action)
            next_state, reward, done, info = self._task.step([action])
            entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
            self._total_steps += 1
            self._state = next_state
            return entry

class DQNAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.replay = config.replay_fn()
        self.actor = DQNActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

        self.num_eigvals = config.num_eigvals
        self.linear_weights = torch.rand(self.num_eigvals*4, requires_grad=True)
        self.target_linear_weights = torch.rand(self.num_eigvals*4, requires_grad=True)
        self.adj_graph_constructed = False
        self.optimizer.add_param_group({'params': self.linear_weights})
        self.optimizer.add_param_group({'params': self.target_linear_weights})

        if config.algo == "pvf":
            self.adj_graph = np.zeros((104, 104))
        elif config.algo == "mrpvf":
            self.adj_graph = np.zeros((4, 104, 104))

        if config.transfer == True:
            self.network.load_state_dict(torch.load(config.load_weights_location))       
            self.target_network.load_state_dict(torch.load(config.load_weights_location))  

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        q = self.network(state)       
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config
        transitions = self.actor.step()
        experiences = []
        for state, action, reward, next_state, done, info in transitions:
            self.record_online_return(info)
            self.total_steps += 1
            reward = config.reward_normalizer(reward)
            experiences.append([state, action, reward, next_state, done])
        self.replay.feed_batch(experiences)

        '''if self.total_steps > self.config.exploration_steps:
            self.adj_graph_constructed = True
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)
            q_next = self.target_network(next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1)[0]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()
            q = self.network(states)
            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step() '''

        obs_map = np.array([np.ones(13)*-1,
        [-1,0,1,2,3,4,-1,5,6,7,8,9,-1],
        [-1,10,11,12,13,14,-1,15,16,17,18,19,-1],
        [-1,20,21,22,23,24,25,26,27,28,29,30,-1],
        [-1,31,32,33,34,35,-1,36,37,38,39,40,-1],
        [-1,41,42,43,44,45,-1,46,47,48,49,50,-1],
        [-1,-1,51,-1,-1,-1,-1,52,53,54,55,56,-1],
        [-1,57,58,59,60,61,-1,-1,-1,62,-1,-1,-1],
        [-1,63,64,65,66,67,-1,68,69,70,71,72,-1],
        [-1,73,74,75,76,77,-1,78,79,80,81,82,-1],
        [-1,83,84,85,86,87,88,89,90,91,92,93,-1],
        [-1,94,95,96,97,98,-1,99,100,101,102,103,-1],
        np.ones(13)*-1])

        # pvf/mrpvf learning
        if self.total_steps > self.config.exploration_steps:

            self.adj_graph_constructed = True

            #visualizing action coverage for different values of exploration timesteps
            '''
            plt.figure(figsize=(20,5))
            plt.subplot(141)
            plt.title('Action 0: Up', fontsize=10)
            temp = self.adj_graph[0]
            temp = np.sum(temp, axis=1)
            temp_obs_map = obs_map.copy()
            for i in range(104):
                temp_obs_map[temp_obs_map==i] = (temp[i]>0).astype(int)
            plt.imshow(temp_obs_map, cmap='Blues')
            plt.axis('off')
            plt.subplot(142)
            plt.title('Action 1: Down', fontsize=10)
            temp = self.adj_graph[1]
            temp = np.sum(temp, axis=1)
            temp_obs_map = obs_map.copy()
            for i in range(104):
                temp_obs_map[temp_obs_map==i] = (temp[i]>0).astype(int)
            plt.imshow(temp_obs_map, cmap='Blues')
            plt.axis('off')
            plt.subplot(143)
            plt.title('Action 2: Left', fontsize=10)
            temp = self.adj_graph[2]
            temp = np.sum(temp, axis=1)
            temp_obs_map = obs_map.copy()
            for i in range(104):
                temp_obs_map[temp_obs_map==i] = (temp[i]>0).astype(int)
            plt.imshow(temp_obs_map, cmap='Blues')
            plt.axis('off')
            plt.subplot(144)
            plt.title('Action 3: Right', fontsize=10)
            temp = self.adj_graph[3]
            temp = np.sum(temp, axis=1)
            temp_obs_map = obs_map.copy()
            for i in range(104):
                temp_obs_map[temp_obs_map==i] = (temp[i]>0).astype(int)
            plt.imshow(temp_obs_map, cmap='Blues')
            plt.axis('off')
            plt.show() '''

            '''plt.figure(figsize=(20,5))
            plt.subplot(141)
            plt.title('Action 0: Up', fontsize=10)
            plt.imshow(self.adj_graph[0], cmap='Blues')
            plt.axis('off')
            plt.subplot(142)
            plt.title('Action 1: Down', fontsize=10)
            plt.imshow(self.adj_graph[1], cmap='Blues')
            plt.axis('off')
            plt.subplot(143)
            plt.title('Action 2: Left', fontsize=10)
            plt.imshow(self.adj_graph[2], cmap='Blues')
            plt.axis('off')
            plt.subplot(144)
            plt.title('Action 3: Right', fontsize=10)
            plt.imshow(self.adj_graph[3], cmap='Blues')
            plt.axis('off')
            plt.show() '''

            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences

            next_state_indices = np.argmax(np.array(next_states), axis=1)
            next_states = self.v[:,next_state_indices,:]
            q_next = (torch.matmul(torch.from_numpy(next_states).float(),self.target_linear_weights)).t()

            if config.control == True:
                q_next = q_next.max(1)[0]  
            elif config.control == False:
                q_next = q_next[torch.arange(q_next.shape[0]), actions]
            terminals = tensor(terminals)
            rewards = tensor(rewards)
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)
            actions = tensor(actions).long()

            state_indices = np.argmax(np.array(states), axis=1)
            states = self.v[:,state_indices,:]
            #print(states[0,0,:][np.nonzero(states[0,0,:] != 0.)])
            #print(self.v.shape)
            #print(self.v[0,:,0])
            #print(np.std(self.v[0,:,72]))
            #print(np.std(self.v[1,:,73]))
            #print(np.std(self.v[2,:,74]))
            #print(np.std(self.v[3,:,75]))
            q = (torch.matmul(torch.from_numpy(states).float(),self.linear_weights)).t()

            #if self.total_steps % 50000 == 0:
            #    print(state_indices, next_state_indices, "\n", q)

            q = q[self.batch_indices, actions]
            loss = (q_next - q).pow(2).mul(0.5).mean()
            
            if config.control == False:
                print(loss)
            
            self.optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
            with config.lock:
                self.optimizer.step()

            self.actor.linear_weights = self.linear_weights


        # pvf graph construction
        elif self.total_steps < self.config.exploration_steps and self.adj_graph_constructed == False and config.algo == "pvf":
            for state, action, reward, next_state, done, info in transitions:
                state_array = np.array(state)
                next_state_array = np.array(next_state)
                if done == True:
                    next_state_array = np.zeros(len(next_state))
                    next_state_array[self.config.goal_location] = 1 
                if np.argmax(state_array) == np.argmax(next_state_array):
                    self.adj_graph[np.argmax(state_array),np.argmax(next_state_array)] = 2
                    self.adj_graph[np.argmax(next_state_array),np.argmax(state_array)] = 2
                else:
                    self.adj_graph[np.argmax(state_array),np.argmax(next_state_array)] = 1
                    self.adj_graph[np.argmax(next_state_array),np.argmax(state_array)] = 1
            
            self.deg_graph = np.diag(np.sum(self.adj_graph, axis=1))
            self.laplacian = self.deg_graph - self.adj_graph
            #print(self.laplacian)
            self.w, self.v = (eigh(self.laplacian, eigvals = (0, self.num_eigvals-1)))
            self.v = np.array(self.v)
            self.v = np.repeat(self.v, 4, axis=1)
            self.v = np.repeat(self.v[np.newaxis, :, :], 4, axis=0)
            for action in range(self.v.shape[0]):
                for col in range(self.v.shape[2]):
                    if col%4 != action:
                        self.v[action,:,col] = 0

            self.actor.v = self.v 

        # mrpvf graph construction
        elif self.total_steps < self.config.exploration_steps and self.adj_graph_constructed == False and config.algo == "mrpvf":
            for state, action, reward, next_state, done, info in transitions:
                state_array = np.array(state)
                next_state_array = np.array(next_state)
                if done == True:
                    next_state_array = np.zeros(len(next_state))
                    next_state_array[self.config.goal_location] = 1 

                if config.decomp_type == 'full_eig':
                    reverse_actions = {0:1, 1:0, 2:3, 3:2}
                    for permissible_action in range(4):
                        if action == permissible_action:
                            self.adj_graph[permissible_action, np.argmax(state_array),np.argmax(next_state_array)] = max(1000, self.adj_graph[permissible_action, np.argmax(state_array),np.argmax(next_state_array)])
                            if done == True:
                                self.adj_graph[reverse_actions[permissible_action], np.argmax(next_state_array), np.argmax(state_array)] = max(1000, self.adj_graph[reverse_actions[permissible_action], np.argmax(next_state_array), np.argmax(state_array)])
                        else:
                            self.adj_graph[permissible_action, np.argmax(state_array),np.argmax(next_state_array)] = max(1, self.adj_graph[permissible_action, np.argmax(state_array),np.argmax(next_state_array)])
                            if done == True:
                                self.adj_graph[reverse_actions[permissible_action], np.argmax(next_state_array), np.argmax(state_array)] = max(1, self.adj_graph[reverse_actions[permissible_action], np.argmax(next_state_array), np.argmax(state_array)])

                else:
                    if np.argmax(state_array) == np.argmax(next_state_array):
                        self.adj_graph[action, np.argmax(state_array),np.argmax(next_state_array)] = 2
                        if config.decomp_type == 'eig':
                            self.adj_graph[action, np.argmax(next_state_array),np.argmax(state_array)] = 2
                    else:
                        self.adj_graph[action, np.argmax(state_array),np.argmax(next_state_array)] = 1
                        if config.decomp_type == 'eig':
                            self.adj_graph[action, np.argmax(next_state_array),np.argmax(state_array)] = 1
            
            if self.total_steps > 9000:
                self.deg_graph = np.zeros((4, 104, 104))
                self.laplacian = np.zeros((4, 104, 104))
                temp_v = np.zeros((4,104,self.num_eigvals*4))
                for action_index in range(4):
                    if config.decomp_type == 'full_eig':
                        # find the P matrix
                        self.deg_graph[action_index] = np.diag(np.sum(self.adj_graph[action_index], axis=1))
                        self.P = np.linalg.solve(self.deg_graph[action_index], self.adj_graph[action_index])
                        # get the Perron vector w
                        self.old_psi_diag = np.random.rand(104)
                        self.old_psi_diag = self.old_psi_diag/np.sum(self.old_psi_diag)
                        self.psi_diag = np.random.rand(104)
                        self.psi_diag = self.psi_diag/np.sum(self.psi_diag)
                        while abs(np.sum(self.old_psi_diag - self.psi_diag))>0.001:
                            self.old_psi_diag = self.psi_diag
                            self.psi_diag = np.dot(self.psi_diag, self.P)
                        # find the laplacian
                        self.psi = np.diag(self.psi_diag)
                        self.laplacian[action_index] = self.psi - (np.matmul(self.psi, self.P)+np.matmul((self.P).transpose(), self.psi))/2
                        # verifying that directed laplacian is symmetric
                        # print((self.laplacian[action_index].transpose() == self.laplacian[action_index]).all())
                        # find the eigen vectors and values into self.w, self.v
                        self.w, self.v = (eigh(self.laplacian[action_index], eigvals = (0, self.num_eigvals-1)))
                        # repeat the self.v =, self.v = , and temp_v[action_index] lines here
                        self.v = np.array(self.v)
                        self.v = np.repeat(self.v, 4, axis=1)
                        temp_v[action_index] = self.v                    
                    else:
                        self.deg_graph[action_index] = np.diag(np.sum(self.adj_graph[action_index], axis=1))
                        self.laplacian[action_index] = self.deg_graph[action_index] - self.adj_graph[action_index]
                        if config.decomp_type == 'eig':
                            self.w, self.v = (eigh(self.laplacian[action_index], eigvals = (0, self.num_eigvals-1)))
                        elif config.decomp_type == 'svd':
                            diag = np.array(np.diagonal(self.adj_graph[action_index]))
                            new_diag = np.array(np.diagonal(self.laplacian[action_index]))
                            new_diag[diag>0] += 1
                            np.fill_diagonal(self.laplacian[action_index],new_diag)
                            u,s,v = svd(self.laplacian[action_index])
                            self.w = s[-config.num_eigvals:]
                            self.v = u[:,-config.num_eigvals:]
                        self.v = np.array(self.v)
                        self.v = np.repeat(self.v, 4, axis=1)
                        temp_v[action_index] = self.v

                self.v = temp_v
                #print(self.laplacian)
                for action in range(self.v.shape[0]):
                    for col in range(self.v.shape[2]):
                        if col%4 != action:
                            self.v[action,:,col] = 0

                self.actor.v = self.v

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            self.target_linear_weights = self.linear_weights

        # saving the model
        if self.total_steps%100000==0 and self.total_steps>0:
            PATH = "log/"+config.tag+"_ntimesteps="+str(self.total_steps)+".pt"
            torch.save(self.network.state_dict(), PATH)
