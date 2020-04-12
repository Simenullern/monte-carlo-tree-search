import os
import torch
from Config import *
from Actor import Actor

class Tournament:
    def __init__(self, m_games_to_play_between_two_player):
        self.m_games_to_play_between_two_player = m_games_to_play_between_two_player

    def load_contesters(self):
        models = {}

        for filename in os.listdir('./models'):
            actor = Actor(BOARD_SIZE, HIDDEN_LAYERS, LEARNING_RATE, ACTIVATION, OPTIMIZER)
            model = actor.net
            model.load_state_dict(torch.load('./models/'+filename))
            model.eval()
            models[filename] = model

        for key in models:
            print(key)
            print(models[key].state_dict())
            print("\n")

        print(models['net_ver0_episode20.pt'] == models['net_ver0_episode30.pt'])

        state = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()
        for key in models:
            o = models[key].forward(state)
            print(o)

        breakpoint()
        return models


if __name__ == '__main__':
    tournament = Tournament(m_games_to_play_between_two_player=20)
    models = tournament.load_contesters()