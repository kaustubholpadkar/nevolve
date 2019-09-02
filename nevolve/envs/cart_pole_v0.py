"""
CartPole Environment of OpenAI Gym.
"""


import gym
import numpy as np
from nevolve.envs import GymEnvironment


gym.logger.set_level(40)


class CartPoleV0(GymEnvironment):
	"""
	Wrapper for CartPole-v0 environment of OpenAI Gym.

	Attributes:
		brain: NeuralNetwork instance
		env: Environment instance
		configuration: Neural Network Configuration
		cls: Environment Class Reference
		dead: bool - if agent is dead
		action: Action to be taken at next step
		fitness: Fitness of agent
		score: Score of agent
		observation: Observation of Environment
	"""

	def __init__(self):
		super().__init__()

		self.env = gym.make("CartPole-v0")
		self.observation = self.env.reset()

		self.configuration = self.get_config()
		self.brain = self.create_brain()

	def think(self):
		"""
		Function for inference based on current observation
		"""
		inputs = self.observation
		inputs[0] += 2.4
		inputs[0] /= 4.8
		inputs[3] /= 3.0
		inputs = np.reshape(inputs, newshape=(1, -1))
		outputs = self.brain.predict(inputs)
		self.action = np.argmax(outputs)

	def act(self):
		"""
		Function to apply the action and get observation and rewards
		"""
		self.observation, reward, self.dead, info = self.env.step(self.action)
		self.fitness += reward
		self.score += reward

	def calculate_fitness(self):
		"""
		Function to set fitness while applying natural selection
		"""
		self.fitness = self.score

	def get_config(self):
		"""
		Get Neural Network Configuration

		Returns:
			list - Neural Network Configuration

			Examples:

				To create Neural Network with 1 hidden layer with input dimension 2, hidden nodes 5 and output dimension 1,
				use following configuration.

				[
					{"input_dim": 2, "output_dim": 5, "activation": "sigmoid"},
					{"input_dim": 5, "output_dim": 1, "activation": "softmax"}
				]

		"""
		input_size = self.env.observation_space.shape[0]
		hidden_size = 10
		output_size = self.env.action_space.n
		return [
			{"input_dim": input_size, "output_dim": hidden_size, "activation": "sigmoid"},
			{"input_dim": hidden_size, "output_dim": output_size, "activation": "softmax"}
		]
