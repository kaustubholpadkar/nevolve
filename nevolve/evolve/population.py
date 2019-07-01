import os
import pickle
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor


class Population:

	def __init__(self, cls, size=500, mutation_rate=0.05, max_workers=8, max_generation=100):
		self.cls = cls
		self.size = size
		self.population = []
		self.best_of_population = None
		self.best_score = 0
		self.best_of_all_generations = None
		self.show_best_of_population = True
		self.fitness_sum = 0

		self.mutation_rate = mutation_rate
		self.show_best = False
		self.last_best_index = 0

		self.generation = 0
		self.max_generation = max_generation

		self.max_workers = max_workers
		self.pool = ThreadPoolExecutor(max_workers=self.max_workers)

		futures = []

		for i in range(self.size):
			future = self.pool.submit(lambda: cls())
			futures.append(future)

		for future in futures:
			self.population.append(future.result())

	def done(self):
		for obj in self.population:
			if not obj.dead:
				return False
		return True

	def update(self):
		if self.show_best_of_population and self.best_of_population and not self.best_of_population.dead:
			self.best_of_population.think()
			self.best_of_population.act()

		for obj in self.population:
			if not obj.dead:
				obj.think()
				obj.act()

	def show(self):
		if not self.show_best_of_population:
			best_index = self.last_best_index
			last_best_obj = self.population[self.last_best_index]
			best_obj = self.population[self.last_best_index]
			for i in range(len(self.population)):
				obj = self.population[i]
				if obj.fitness > best_obj.fitness:
					best_obj = obj
					best_index = i
				if obj.dead:
					obj.close()
				else:
					if not self.show_best:
						obj.show()

			if self.show_best and not best_obj.dead:
				if best_index != self.last_best_index:
					last_best_obj.close()
					self.last_best_index = best_index
				best_obj.show()

		if self.show_best_of_population and self.best_of_population and not self.best_of_population.dead:
			self.best_of_population.show()

	def set_best_of_population(self):

		best_obj = self.population[self.last_best_index]
		for i in range(len(self.population)):
			obj = self.population[i]
			if obj.fitness > best_obj.fitness:
				best_obj = obj
			if obj.dead:
				obj.close()

		self.best_score = best_obj.fitness
		self.best_of_population = best_obj.clone()

	def select_parent(self):
		rand = np.random.random() * self.fitness_sum
		summation = 0

		for i in range(len(self.population)):
			obj = self.population[i]
			summation += obj.fitness
			if summation > rand:
				return obj.clone()
		return self.population[0].clone()

	def mutate(self):
		for obj in self.population:
			obj.mutate(self.mutation_rate)

	def calculate_fitness(self):
		for obj in self.population:
			obj.calculate_fitness()

	def calculate_fitness_sum(self):
		self.fitness_sum = 0
		for obj in self.population:
			self.fitness_sum += obj.fitness

	def natural_selection(self):
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
		for obj in self.population:
			obj.close()

	def explore(self, index):

		agent = self.population[index]

		while not agent.dead:
			agent.think()
			agent.act()

	def evolve(self, show_best=False, checkpoint=False, checkpoint_dir=".", checkpoint_prefix="", verbose=True):
		best_thread = None

		for i in range(self.max_generation):
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
		while self.best_of_population and not self.best_of_population.dead:
			self.best_of_population.think()
			self.best_of_population.act()
			self.best_of_population.show()

		if self.best_of_population:
			self.best_of_population.close()

	def save_population(self, path):
		data = [obj.get_brain() for obj in self.population]
		with open(path, 'wb') as outfile:
			pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

	def load_population(self, path):
		with open(path, 'rb') as infile:
			data = pickle.load(infile)

		for i in range(len(data)):
			self.population[i].set_brain(data[i])

	def save_best_(self, path):
		with open(path, 'wb') as outfile:
			pickle.dump(self.best_of_population.get_brain(), outfile, pickle.HIGHEST_PROTOCOL)

	def load_best_(self, path):
		self.best_of_population = self.cls()
		with open(path, 'rb') as infile:
			data = pickle.load(infile)
			self.best_of_population.set_brain(data)
