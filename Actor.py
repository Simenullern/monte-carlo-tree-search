import torch
import Config
from Hex import Hex
import Utils


class Actor:
    def __init__(self, board_size, hidden_layers, learning_rate, activation, optimizer):
        modules = []
        modules.append(torch.nn.Linear(Hex.get_number_of_cells(board_size) + 1, hidden_layers[0]))  # +1 from playerID
        modules.append(Config.ACTIVATIONS[activation]())

        for i in range(0, len(hidden_layers) - 1):
            modules.append(torch.nn.Linear((hidden_layers[i]), hidden_layers[i + 1]))
            modules.append(Config.ACTIVATIONS[activation]())

        modules.append(torch.nn.Linear(hidden_layers[-1], Hex.get_number_of_cells(board_size)))
        modules.append(torch.nn.Softmax()) # Not use? # cross entropy loss: The input is expected to contain raw, unnormalized scores for each class.

        self.net = torch.nn.Sequential(*modules)
        self.criterion = torch.nn.MSELoss()  # Cross entropy. BCE Loss
        self.optimizer = Config.OPTIMIZERS[optimizer](self.net.parameters(), lr=learning_rate)

    def forward(self, X):
        self.net.eval()
        return self.net(X)

    def train(self, replay_buffer, replay_buffer_max_size, replay_buffer_minibatch_size):
        training_set = replay_buffer
        if len(replay_buffer) > replay_buffer_max_size:
            training_set = replay_buffer[:replay_buffer_max_size]
        training_set = Utils.shuffle(training_set)

        self.net.train()
        for example in training_set[:replay_buffer_minibatch_size]:
            self.optimizer.zero_grad()
            pred = self.net(example[0])
            visit_distr = torch.tensor(example[1])
            loss = self.criterion(pred, visit_distr)
            loss.backward()
            self.optimizer.step()
        return self

    def save(self, episode):
        state_dict = self.net.state_dict()
        torch.save(state_dict, './models/net_after_episode_'+str(episode)+".pt")
