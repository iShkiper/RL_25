from gymnasium import Env
from gymnasium import spaces
from numpy.random import choice

class customMDP(Env):
	metadata = {'render.modes': []}

	def __init__(self, p_sale):
        # число состояний
		self.observation_space = spaces.Discrete(3)
		self.state = choice(list(range(0, 2)))
		self.p_sale = p_sale        
		# 4 действия
		self.action_space = spaces.Discrete(2)

        # динамика среды
		self.transition_matrix = [
        # Ничего не делать
		[
			[1, 0, 0],
			[self.p_sale, 1 - self.p_sale, 0],
			[0, self.p_sale, 1 - self.p_sale],
		],
        # Пополнить
		[
			[self.p_sale, 1 - self.p_sale, 0],
			[0, self.p_sale, 1 - self.p_sale],
			# в состоянии 2 пополнение невозможно
		],
		]
        
		self.reward_matrix = [
        # Ничего не делать
		[
			[0, 0, 0],
			[1, 0, 0],
			[0, 1, 0],
		],
        # Пополнить
		[
			[1, 0, 0],
			[0, 1, 0],
			# в состоянии 2 пополнение невозможно
		],
		]

		self.P = {}
		self.setP()
        
            # формирование матрицы с динамикой переходов
	def setP(self):
		for state in range(self.observation_space.n):
			self.P[state] = {}
			for act in range(self.action_space.n):
				probs = self.transition_matrix[act][state]
				self.P[state][act] = []                 
				for ind, prob in enumerate(probs):
					new_state = ind
					reward = self.reward_matrix[act][state][new_state]
					if prob != 0:
						self.P[state][act].append((prob, new_state, reward, False))
        
	def reset(self):
		self.state = choice(list(range(0, 2)))
		return self.state, ''
    
	def step(self, action):
		probs = self.transition_matrix[action][self.state] 
		new_state = choice(range(self.observation_space.n), 1, p=probs)[0]
		reward = self.reward_matrix[action][self.state][new_state]
		self.state = new_state
		return self.state, reward, False, '', ''
      