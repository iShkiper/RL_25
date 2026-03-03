from gymnasium import Env
from gymnasium import spaces
from numpy.random import choice
from numpy import unravel_index
from numpy import ravel_multi_index

class GridWorld4by4(Env):
	metadata = {'render.modes': []}

	def __init__(self):
		# размер клеточного мира 4 на 4 клетки, 16 состояний
		self.size = 4        
		self.observation_space = spaces.Discrete(16)
		self.state = 0
		self.done = False
        
		# 4 действия
		self.action_space = spaces.Discrete(4)

		self.terminal_reward = {0: 0, 15: 0}
		self.default_reward = -1
        # словарь для хранения динамики среды
		self.P = {} 
		self.setP()
        
	def reset(self):
		self.state = choice(list(range(1, 15)))
		self.done = False
		return self.state, ''
    
        # формирование матрицы с динамикой переходов
	def setP(self):
		for state in range(self.observation_space.n):
			self.P[state] = {}
			for act in range(self.action_space.n):
				self.state = state
				self.done = False
				new_state, reward, _, _, _ = self.step(act)
				self.P[state][act] = [(1.0, new_state, reward, self._isTerminal(state))]

	def _isTerminal(self, state):
		return state == 0 or state == 15
    
	def step(self, action):
		if self._isTerminal(self.state):
			return self.state, 0, True, False, ''
		reward = self._get_reward()
		self._take_action(action)
		return self.state, reward, self.done, False, ''

	def _take_action(self, action):
		state = unravel_index(self.state, (self.size, self.size))

		if action == 0: #влево
			next_state = (state[0], state[1]-1)
		elif action == 1: #вниз
			next_state = (state[0]+1, state[1])
		elif action == 2: #вправо
			next_state = (state[0], state[1]+1)
		else: #вверх
			next_state = (state[0]-1, state[1])
		
		# обновить состояния, если мы не вышли за границы
		if (0 <= next_state[0] < self.size) and (0 <= next_state[1] < self.size):
			self.state = ravel_multi_index(next_state, (self.size, self.size))          

	def _get_reward(self):
		if self.state in self.terminal_reward:
			return self.terminal_reward[self.state]
		else:
			return self.default_reward