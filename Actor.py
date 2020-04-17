import torch
import Config
from Hex import Hex
import Utils
import numpy as np


class Actor:
    def __init__(self, board_size, hidden_layers, learning_rate, activation, optimizer):
        modules = []
        modules.append(torch.nn.Linear(Hex.get_number_of_cells(board_size)+1+2+4 , hidden_layers[0]))  # +1 from playerID. Then frac of free and taken cells. Then walls.
        modules.append(Config.ACTIVATIONS[activation]())

        for i in range(0, len(hidden_layers) - 1):
            modules.append(torch.nn.Linear((hidden_layers[i]), hidden_layers[i + 1]))
            modules.append(Config.ACTIVATIONS[activation]())

        modules.append(torch.nn.Linear(hidden_layers[-1], Hex.get_number_of_cells(board_size)))
        #modules.append(torch.nn.Softmax()) # Not use? The input is expected to contain raw, unnormalized scores for each class.

        self.net = torch.nn.Sequential(*modules)
        self.criterion = self.cross_entropy  # Cross entropy. BCE Loss # MSELoss()
        self.optimizer = Config.OPTIMIZERS[optimizer](self.net.parameters(), lr=learning_rate)

    def cross_entropy(self, pred, soft_targets):
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


    def forward(self, X):
        self.net.eval()
        return self.net(X)

    def train(self, replay_buffer, new_examples_count, replay_buffer_max_size, replay_buffer_minibatch_size):
        new_examples = replay_buffer[-new_examples_count:]
        old_examples = replay_buffer[:-new_examples_count]
        if len(old_examples) > replay_buffer_max_size:  # only keep the latest most entries
            old_examples = old_examples[:-replay_buffer_max_size]
        old_examples_shuffled = Utils.shuffle(old_examples)
        training_set = new_examples
        if len(old_examples) > 0:
            training_set = (new_examples + old_examples_shuffled)[:replay_buffer_minibatch_size]
        self.net.train()
        for example in training_set:
            self.optimizer.zero_grad()
            pred = self.net(example[0]).reshape(1, -1)
            argmax = torch.tensor(np.argmax(example[1])).reshape(1)
            distr = torch.tensor(example[1])
            loss = self.criterion(pred, distr)
            loss.backward()
            self.optimizer.step()
        return self

    def save(self, board_size, episode):
        state_dict = self.net.state_dict()
        torch.save(state_dict, './models/boardsize_' + str(board_size) + '/net_after_episode_'+str(episode)+".pt")
