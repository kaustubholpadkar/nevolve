"""
Module for Neuro-Evolution implementation.
"""


import os
import pickle
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class Population:
	"""
	Class for Neuro-Evolution

	Args:
		cls: class for the Environment
		size: Population size
		mutation_rate: Mutation Rate
		max_workers: Maximum Workers

	Attributes:
		cls: class for the Environment
		size: Population size
		population: list
		best_of_population: instance of the best agent in the population
		best_score: score of the best agent in the population
		fitness_sum: sum of the fitness of all agents in the population
		mutation_rate: Mutation Rate
		generation: current generation number
		max_workers: Maximum Workers
		pool: instance of ThreadPoolExecutor
	"""

	def __init__(self, cls, size=500, mutation_rate=0.05, max_workers=8):

		self.cls = cls
		self.size = size
		self.population = []
		self.best_of_population = None
		self.best_score = 0
		self.fitness_sum = 0

		self.mutation_rate = mutation_rate

		self.generation = 0

		self.max_workers = max_workers
		self.pool = ThreadPoolExecutor(max_workers=self.max_workers)

		futures = []

		for i in range(self.size):
			future = self.pool.submit(lambda: cls())
			futures.append(future)

		for future in futures:
			self.population.append(future.result())

	def set_best_of_population(self):
		"""
		Set the Best of Population
		"""
		best_obj = self.population[0]
		for i in range(len(self.population)):
			obj = self.population[i]
			if obj.fitness > best_obj.fitness:
				best_obj = obj
			if obj.dead:
				obj.close()

		self.best_score = best_obj.fitness
		self.best_of_population = best_obj.clone()

	def select_parent(self):
		"""
		Function to select parent (Selection Algorithm based on Genetic Algorithm)

		Returns:
			instance of Environment
		"""
		rand = np.random.random() * self.fitness_sum
		summation = 0

		for i in range(len(self.population)):
			obj = self.population[i]
			summation += obj.fitness
			if summation > rand:
				return obj.clone()
		return self.population[0].clone()

	def mutate(self):
		"""
		Mutate the Population
		"""
		for obj in self.population:
			obj.mutate(self.mutation_rate)

	def calculate_fitness(self):
		"""
		Calculate Fitness for Population
		"""
		for obj in self.population:
			obj.calculate_fitness()

	def calculate_fitness_sum(self):
		"""
		Calculate Fitness Sum
		"""
		self.fitness_sum = 0
		for obj in self.population:
			self.fitness_sum += obj.fitness

	def natural_selection(self):
		"""
		Natural Selection Algorithm to create Next Generation Population
		"""
		new_population = []

		self.set_best_of_population()
		self.calculate_fitness_sum()

		for i in range(len(self.population)):
			child = self.select_parent()
			child.mutate(self.mutation_rate)
			new_population.append(child)

		self.population = new_population
		self.generation += 1

	def close(self):
		"""
		Function to clean up the Population
		"""
		for obj in self.population:
			obj.close()

	def explore(self, index):
		"""
		Explore the Environment

		Args:
			index: index of agent to explore
		"""
		agent = self.population[index]

		while not agent.dead:
			agent.think()
			agent.act()

	def evolve(self, max_generation=100, show_best=False, checkpoint=False, checkpoint_dir=".", checkpoint_prefix="", verbose=True):
		"""
		Function to start Neuro-Evolution

		Args:
			show_best: show best population
			checkpoint: checkpoint flag
			checkpoint_dir: checkpoint directory
			checkpoint_prefix: prefix to checkpoint files
			verbose: print intermediate messages
			max_generation: Maximum number of Generations
		"""
		best_thread = None

		for i in range(max_generation):
			if show_best:
				best_thread = threading.Thread(target=self.display_best)
				best_thread.start()

			futures = []

			for index in range(len(self.population)):
				future = self.pool.submit(self.explore, index)
				futures.append(future)

			for future in futures:
				future.result()

			if show_best:
				best_thread.join()

			if checkpoint:
				file_path = os.path.join(checkpoint_dir, "{}_generation_{}.pkl".format(checkpoint_prefix, i+1))
				self.save_population(file_path)

			self.calculate_fitness()
			self.natural_selection()

			if verbose:
				print("Generation: {} | Best Fitness: {}".format(i+1, self.best_score))

		self.close()

	def display_best(self):
		"""
		Function to Display Best Agent
		"""
		while self.best_of_population and not self.best_of_population.dead:
			self.best_of_population.think()
			self.best_of_population.act()
			self.best_of_population.show()

		if self.best_of_population:
			self.best_of_population.close()

	def save_population(self, path):
		"""
		Save the Population to File

		Args:
			path: path where to store file
		"""
		data = [obj.get_brain() for obj in self.population]
		with open(path, 'wb') as outfile:
			pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

	def load_population(self, path):
		"""
		Load the Population from File

		Args:
			path: path from where to load population
		"""
		with open(path, 'rb') as infile:
			data = pickle.load(infile)

		for i in range(len(data)):
			self.population[i].set_brain(data[i])

	def save_best_(self, path):
		"""
		Save the Best Population to File

		Args:
			path: path where to store file
		"""
		with open(path, 'wb') as outfile:
			pickle.dump(self.best_of_population.get_brain(), outfile, pickle.HIGHEST_PROTOCOL)

	def load_best_(self, path):
		"""
		Load the Best Population from File

		Args:
			path: path from where to load agent
		"""
		self.best_of_population = self.cls()
		with open(path, 'rb') as infile:
			data = pickle.load(infile)
			self.best_of_population.set_brain(data)
