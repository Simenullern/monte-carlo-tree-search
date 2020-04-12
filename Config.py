import torch

# HEX
BOARD_SIZE = 3

VISUALIZE_MOVES = False
NUM_EPISODES = 30
NUM_OF_SIMULATIONS = 10
EXPLORATION_BONUS_C = 1
SAVE_PARAMS_EVERY_NTH_EPISODE = 10
STARTING_PLAYER = 1


HIDDEN_LAYERS = [16, 12]
LEARNING_RATE = 10
ACTIVATION = 'relu'  #'sigmoid', 'tanh', 'relu'
OPTIMIZER = 'adam'  #adagrad, sgd, rmsprop, 'adam'
RANDOM_MINIBATCH_SIZE = 10

M_GAMES_TO_PLAY_IN_TOPP = 20

ACTIVATIONS = {
    'linear': torch.nn.Linear,
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
