import gym
import numpy as np
from nevolve.envs import GymEnvironment


gym.logger.set_level(40)


class MountainCarV0(GymEnvironment):

	def __init__(self):
		super().__init__()

		self.env = gym.make("MountainCar-v0")
		self.observation = self.env.reset()

		self.configuration = self.get_config()
		self.brain = self.create_brain()

	def think(self):
		inputs = self.observation
		inputs[0] += 1.2
		inputs[0] /= 1.8
		inputs[1] += 0.07
		inputs[1] /= 0.14
		inputs = np.reshape(inputs, newshape=(1, -1))
		outputs = self.brain.predict(inputs)
		self.action = np.argmax(outputs)

	def act(self):
		self.observation, reward, self.dead, info = self.env.step(self.action)
		self.fitness = max(self.observation[0], self.fitness)
		self.score = self.fitness

	def calculate_fitness(self):
		self.fitness = self.score

	def get_config(self):
		input_size = self.env.observation_space.shape[0]
		hidden_size = 10
		output_size = self.env.action_space.n
		return [
			{"input_dim": input_size, "output_dim": hidden_size, "activation": "sigmoid"},
			{"input_dim": hidden_size, "output_dim": output_size, "activation": "softmax"}
		]
