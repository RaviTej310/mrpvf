import sys
from contextlib import closing
from six import StringIO
import gym
from gym import error, spaces, utils
from gym.envs.toy_text import discrete
import numpy as np


class FourRooms(gym.Env):

	def __init__(self, goal_location):
		layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
		self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
		
		# Four possible actions
		# 0: UP
		# 1: DOWN
		# 2: LEFT
		# 3: RIGHT
		#self.action_space = np.array([0, 1, 2, 3])
		self.observation_space = np.zeros(np.sum(self.occupancy == 0))
		self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]
		self.action_space = spaces.Discrete(4)

		# Random number generator
		self.rng = np.random.RandomState(1234)

		self.tostate = {}
		statenum = 0
		for i in range(13):
			for j in range(13):
				if self.occupancy[i,j] == 0:
					self.tostate[(i,j)] = statenum
					statenum += 1
		self.tocell = {v:k for k, v in self.tostate.items()}


		self.goal = goal_location #62 # East doorway
		self.init_states = list(range(self.observation_space.shape[0]))
		self.init_states.remove(self.goal)


	def render(self, show_goal=True):
		current_grid = np.array(self.occupancy)
		current_grid[self.current_cell[0], self.current_cell[1]] = -1
		if show_goal:
			goal_cell = self.tocell[self.goal]
			current_grid[goal_cell[0], goal_cell[1]] = -1
		return current_grid

	def reset(self):
		state = self.rng.choice(self.init_states)
		self.current_cell = self.tocell[state]
		temp = np.zeros(len(self.observation_space))
		temp[state] = 1
		return temp

	def check_available_cells(self, cell):
		available_cells = []

		for action in range(self.action_space.n):
			next_cell = tuple(cell + self.directions[action])

			if not self.occupancy[next_cell]:
				available_cells.append(next_cell)

		return available_cells

	def seed(self, seed=None):
		"""
		Seed must be a non-negative 
		"""
		dummy = 0

	def step(self, action):
		'''
		Takes a step in the environment with 49/50 probability. And takes a step in the
		other directions with probability 1/50 with all of them being equally likely.
		'''

		next_cell = tuple(self.current_cell + self.directions[action])

		if not self.occupancy[next_cell]:

			if self.rng.uniform() < 1/20: #1/50:
				available_cells = self.check_available_cells(self.current_cell)
				while len(available_cells)<4:
					available_cells.append(self.current_cell)
					#print(available_cells)
				self.current_cell = available_cells[self.rng.randint(len(available_cells))]
				reward = -1

			else:
				self.current_cell = next_cell
				reward = -1

		else:
			reward = -1

		state = self.tostate[self.current_cell]

		# When goal is reached, it is done
		done = state == self.goal
		if done:
			reward = 20

		temp = np.zeros(len(self.observation_space))
		temp[state] = 1

		return temp, reward, done, {'state': "dummy"}
		#return temp, float(done), done, {'state': "dummy"}











