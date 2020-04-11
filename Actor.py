import torch
import Config
from Hex import Hex

class Actor:
    def __init__(self, board_size, hidden_layers, learning_rate, activation, optimizer):
        modules = []
        modules.append(torch.nn.Linear(Hex.get_number_of_cells(board_size) + 1, hidden_layers[0]))
        modules.append(Config.ACTIVATIONS[activation]())

        for i in range(0, len(hidden_layers) - 1):
            modules.append(torch.nn.Linear((hidden_layers[i]), hidden_layers[i + 1]))
            modules.append(Config.ACTIVATIONS[activation]())

        modules.append(torch.nn.Linear(hidden_layers[-1], Hex.get_number_of_cells(board_size)))
        modules.append(torch.nn.Softmax())

        self.net = torch.nn.Sequential(*modules)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = Config.OPTIMIZERS[optimizer](self.net.parameters(), lr=learning_rate)

    def forward(self, X):
        return self.net(X)