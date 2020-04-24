import torch
from Config import ACTIVATIONS, OPTIMIZERS, NUMBER_OF_ENGINEERED_FEATURES_BEYOND_PLAYER_ID_AND_CELLS
from Hex import Hex
import Utils


class Actor:
    def __init__(self, board_size, hidden_layers, learning_rate, activation, optimizer):
        modules = []
        modules.append(torch.nn.Linear(Hex.get_number_of_cells(board_size)+1 +
                                       NUMBER_OF_ENGINEERED_FEATURES_BEYOND_PLAYER_ID_AND_CELLS,
                                       hidden_layers[0]))
        modules.append(ACTIVATIONS[activation]())

        for i in range(0, len(hidden_layers) - 1):
            modules.append(torch.nn.Linear((hidden_layers[i]), hidden_layers[i + 1]))
            modules.append(ACTIVATIONS[activation]())

        modules.append(torch.nn.Linear(hidden_layers[-1], Hex.get_number_of_cells(board_size)))

        self.net = torch.nn.Sequential(*modules)
        self.criterion = self.cross_entropy_loss
        self.optimizer = OPTIMIZERS[optimizer](self.net.parameters(), lr=learning_rate)

    @staticmethod
    def cross_entropy_loss(pred, soft_targets):
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

    def forward(self, X):
        self.net.eval()
        return self.net(X)

    def train(self, replay_buffer, new_examples_count, replay_buffer_max_size, replay_buffer_minibatch_size):
        new_examples = replay_buffer[-new_examples_count:]
        old_examples = replay_buffer[:-new_examples_count]
        if len(old_examples) > replay_buffer_max_size:
            old_examples = old_examples[:-replay_buffer_max_size]
        old_examples_shuffled = Utils.shuffle(old_examples)
        training_set = new_examples
        if len(old_examples) > 0:
            training_set = (new_examples + old_examples_shuffled)[:replay_buffer_minibatch_size]

        self.net.train()
        for example in training_set:
            self.optimizer.zero_grad()
            pred = self.net(example[0]).reshape(1, -1)
            distr = torch.tensor(example[1])
            loss = self.criterion(pred, distr)
            loss.backward()
            self.optimizer.step()
        return self

    def save(self, board_size, episode):
        state_dict = self.net.state_dict()
        torch.save(state_dict, './models/boardsize_' + str(board_size) + '/net_after_episode_'+str(episode)+".pt")

    @staticmethod
    def load_model(model_path, size, hidden_layers, learning_rate, activation, optimizer):
        actor = Actor(size, hidden_layers, learning_rate, activation, optimizer)
        model = actor.net
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
