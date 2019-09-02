from nevolve.neuro.dnn import DNN


config = [
	{"input_dim": 1, "output_dim": 1, "activation": "linear"}
]

model = DNN(config=config)

assert isinstance(model, DNN)

assert isinstance(model.weights, list) and len(model.weights) >= 1
assert isinstance(model.biases, list) and len(model.biases) >= 1
assert isinstance(model.activations, list) and len(model.activations) >= 1

assert len(model.weights) == len(model.biases) == len(model.activations)

assert model.predict([0.]) == model.biases[0]

config = [
	{"input_dim": 10, "output_dim": 5, "activation": "sigmoid"},
	{"input_dim": 5, "output_dim": 1, "activation": "sigmoid"}
]

model = DNN(config=config)

assert isinstance(model, DNN)

assert isinstance(model.weights, list) and len(model.weights) >= 1
assert isinstance(model.biases, list) and len(model.biases) >= 1
assert isinstance(model.activations, list) and len(model.activations) >= 1
