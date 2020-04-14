import torch

# HEX
BOARD_SIZE = 4

VISUALIZE_MOVES = False
NUM_EPISODES = 50
NUM_OF_SIMULATIONS = 500
EXPLORATION_BONUS_C = 1
SAVE_PARAMS_EVERY_NTH_EPISODE = 10
STARTING_PLAYER = 0

HIDDEN_LAYERS = [64, 64]
LEARNING_RATE = 0.005  # If too high then probability inputs might contain nan
ACTIVATION = 'tanh'  #'sigmoid', 'tanh', 'relu', 'linaer'
OPTIMIZER = 'adam'  #adagrad, sgd, rmsprop, 'adam'

REPLAY_BUFFER_MAX_SIZE = 500
REPLAY_BUFFER_MINIBATCH_SIZE = 50

EPSILON = 0.5

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

MAP_NET_INPUT_FORMAT = {
    2: -1
}
