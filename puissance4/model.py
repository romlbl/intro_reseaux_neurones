import numpy as np

param_model = {
    "W1": np.random.rand(64, 42),
    "b1": np.zeros(64),
    "W2": np.random.rand(7, 64),
    "b2": np.zeros(7)
}

param_model_2 = {
    "W1": np.random.rand(64, 42),
    "b1": np.zeros(64),
    "W2": np.random.rand(32, 64),
    "b2": np.zeros(32),
    "W3": np.random.rand(7,32),
    "b3": np.zeros(7)
}


def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def forward_pass(plateau):
    z1 = np.dot(param_model["W1"],plateau) + param_model["b1"]
    n1 = relu(z1)
    z2 = np.dot(param_model["W2"],n1) + param_model["b2"]
    n2 = softmax(z2)

    return z1, n1, z2, n2

def forward_pass_2(plateau):
    z1 = np.dot(param_model_2["W1"],plateau) + param_model_2["b1"]
    n1 = relu(z1)
    z2 = np.dot(param_model_2["W2"],n1) + param_model_2["b2"]
    n2 = relu(z2)
    z3 = np.dot(param_model_2["W3"],n2) + param_model_2["b3"]
    n3 = softmax(z3)

    return z1, n1, z2, n2, z3, n3
