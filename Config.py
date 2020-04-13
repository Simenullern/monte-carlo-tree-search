import torch

# HEX
BOARD_SIZE = 3

VISUALIZE_MOVES = False
NUM_EPISODES = 100
NUM_OF_SIMULATIONS = 100
EXPLORATION_BONUS_C = 1
SAVE_PARAMS_EVERY_NTH_EPISODE = 50
STARTING_PLAYER = "RANDOM"

HIDDEN_LAYERS = [32, 32, 32]
LEARNING_RATE = 0.01  # If too high then probability inputs contain nan
ACTIVATION = 'sigmoid'  #'sigmoid', 'tanh', 'relu', 'linaer'
OPTIMIZER = 'adam'  #adagrad, sgd, rmsprop, 'adam'

RANDOM_MINIBATCH_SIZE = 20
EPSILON = 0.25

M_GAMES_TO_PLAY_IN_TOPP = 100

ACTIVATIONS = {
    'linear': torch.nn.Identity,
    'sigmoid': torch.nn.Sigmoid,
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU
}

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'rmsprop': torch.optim.RMSprop
}
