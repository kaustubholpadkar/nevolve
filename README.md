# Nevolve - Neuro-Evolution for Humans
Neuro-Evolution Library for Reinforcement Learning.


## Getting started: 15 seconds to Nevolve

The core class of Nevolve is a __Population__, a way to organize whole Neuro-Evolution process.

Here is the `Population` class:

```python
from nevolve.evolve.population import Population
```

Here is the wrapper on `CartPole` environment from OpenAi Gym:

```python
from nevolve.envs.cart_pole_v0 import CartPoleV0
```

Create Population instance.

```python
model = Population(
    cls=CartPoleV0, 
    size=100, 
    mutation_rate=0.05, 
    max_workers=10
)
```

Start Evolution:

```python
model.evolve(
    max_generation=100,
    show_best=True
)
```

Results:
```
Generation: 1 | Best Fitness: 27.0
Generation: 2 | Best Fitness: 21.0
Generation: 3 | Best Fitness: 41.0
Generation: 4 | Best Fitness: 49.0
Generation: 5 | Best Fitness: 62.0
Generation: 6 | Best Fitness: 48.0
Generation: 7 | Best Fitness: 200.0
Generation: 8 | Best Fitness: 200.0
Generation: 9 | Best Fitness: 200.0
Generation: 10 | Best Fitness: 200.0

```

![](bestcartpole.gif)

Once your model looks good, save the best model with `.save_best_()`:

```python
model.save_best_(path="best_cart_pole.pkl")
```

For a more in-depth tutorial about Nevolve, you can check out:

- [Applying Neuro-Evolution on Sample Environments - CartPole, MountainCar & BipedalWalkerV2](https://keras.io/getting-started/sequential-model-guide)
- [Create your own Environment wrappers for Neuro-Evolution - Pong Game.](https://keras.io/getting-started/functional-api-guide)
- [Saving and Loading Populations.](https://keras.io/getting-started/functional-api-guide)


------------------

## Installation

- **Install Nevolve from PyPI (recommended):**

Note: These installation steps assume that you are on a Linux or Mac environment.
If you are on Windows, you will need to remove `sudo` to run the commands below.

```sh
sudo pip install nevolve
```

If you are using a virtualenv, you may want to avoid using sudo:

```sh
pip install nevolve
```

- **Alternatively: install Nevolve from the GitHub source:**

First, clone Nevolve using `git`:

```sh
git clone https://github.com/kaustubholpadkar/nevolve.git
```

 Then, `cd` to the nevolve folder and run the install command:
```sh
cd nevolve
sudo python setup.py install
```

------------------

## Support

You can ask questions and join the development discussion:

- On the [Nevolve Google group](https://groups.google.com/forum/#!forum/nevolve).
- On the [Nevolve Slack channel](https://nevolve.slack.com). Use [this link](https://nevolve-slack-invitation.herokuapp.com/) to request an invitation to the channel.

You can also post **bug reports and feature requests** (only) in [GitHub issues](https://github.com/kaustubholpadkar/nevolve/issues).


------------------

## License

Nevolve is GNU licensed, as found in the LICENSE file.


------------------