from nevolve.envs.cart_pole_v0 import CartPoleV0
# from nevolve.envs.bipedal_walker_v2 import BipedalWalkerV2
# from nevolve.envs.mountain_car_v0 import MountainCarV0
from nevolve.evolve.population import Population


MUTATION_RATE = 0.05
POPULATION_SIZE = 100
MAX_GENERATION = 10

MAX_WORKERS = 20


population = Population(cls=CartPoleV0, size=POPULATION_SIZE, mutation_rate=MUTATION_RATE, max_generation=MAX_GENERATION)
# population = Population(cls=BipedalWalkerV2, size=POPULATION_SIZE, mutation_rate=MUTATION_RATE, max_generation=MAX_GENERATION, max_workers=MAX_WORKERS, show_best=True)
# population = Population(cls=MountainCarV0, size=POPULATION_SIZE, mutation_rate=MUTATION_RATE, max_generation=MAX_GENERATION, max_workers=MAX_WORKERS, show_best=True)
population.evolve(show_best=True)
population.display_best()
