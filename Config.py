import torch

# HEX
#BOARD_SIZE = 3

VISUALIZE_MOVES = False
NUM_EPISODES = 200
NUM_OF_SIMULATIONS = 500
EXPLORATION_BONUS_C = 10   # 1 ## previous runs has 100
SAVE_PARAMS_EVERY_NTH_EPISODE = 50
STARTING_PLAYER = 0

HIDDEN_LAYERS = [48, 48]
LEARNING_RATE = 0.001
ACTIVATION = 'tanh'  #'sigmoid', 'tanh', 'relu', 'linaer'
OPTIMIZER = 'adam'  #adagrad, sgd, rmsprop, 'adam'

REPLAY_BUFFER_MAX_SIZE = 2000
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

