import gym
import numpy as np
from nevolve.envs import GymEnvironment


gym.logger.set_level(40)


class BipedalWalkerV2(GymEnvironment):
	"""
	Wrapper for BipedalWalker-v2 environment of OpenAI Gym
	"""

	def __init__(self):
		"""
		Constructor of BipedalWalkerV2
		"""
		super().__init__()

		self.env = gym.make("BipedalWalker-v2")
		self.observation = self.env.reset()

		self.configuration = self.get_config()
		self.brain = self.create_brain()

	def think(self):
		"""
		Function for inference based on current observation
		"""
		inputs = self.observation
		inputs[0] /= 2 * 3.1415

		inputs[2] += 1
		inputs[2] /= 2

		inputs[3] += 1
		inputs[3] /= 2

		inputs = np.reshape(inputs, newshape=(1, -1))
		outputs = self.brain.predict(inputs)[0]
		self.action = outputs

	def act(self):
		"""
		Function to apply the action and get observation and rewards
		"""
		self.observation, reward, self.dead, info = self.env.step(self.action)
		if reward == -100:
			self.dead = True
		elif reward < 0:
			self.fitness += reward
			self.score = self.fitness
		else:
			self.fitness += reward
			self.score = self.fitness

		if self.fitness < -20:
			self.dead = True

	def calculate_fitness(self):
		"""
		Function to set fitness while applying natural selection
		"""
		if self.score < 0:
			self.fitness = 0.0
		else:
			self.fitness = self.score * self.score

	def get_config(self):
		"""
		Get Neural Network Configuration
		:return: list
		"""
		input_size = self.env.observation_space.shape[0]
		hidden_size = 16
		output_size = self.env.action_space.shape[0]
		return [
			{"input_dim": input_size, "output_dim": hidden_size, "activation": "relu"},
			{"input_dim": hidden_size, "output_dim": hidden_size, "activation": "relu"},
			{"input_dim": hidden_size, "output_dim": output_size, "activation": "tanh"}
		]
