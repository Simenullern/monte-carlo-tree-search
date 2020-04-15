import os
from Config import *
from Actor import Actor
import Utils
from Hex import Hex
from GameController import GameController
from itertools import cycle


def load_model(model_path, size):
    actor = Actor(size, HIDDEN_LAYERS, LEARNING_RATE, ACTIVATION, OPTIMIZER)
    model = actor.net
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def make_move(model, current_state, state_with_player):
    softmax_distr = model.forward(state_with_player).detach().numpy()
    softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)

    action = Utils.make_max_move_from_distribution(softmax_distr_re_normalized, BOARD_SIZE)
    return action


if __name__ == '__main__':
    size = 3
    game = Hex(size=size)
    model = load_model('./models/boardsize_'+str(size) +'/net_after_episode_100.pt', size)
    gameController = GameController(game, visualize=True)
    start_state = gameController.get_game_state()

    while True:
        for game in range(1, 11):
            gameController.reset_game()
            gameController.visualize()

            GAME_CYCLE = cycle(['Player1', 'Player2']) if game % 2 == 0 else cycle(['Player2', 'Player1'])
            # Player 1 is YOU

            for player in GAME_CYCLE:
                if player == 'Player1':
                    action = input("What is your action human? e.g. type 0 0 ")
                    action = tuple(map(int, action.split(' ')))
                    gameController.make_move(action, player)
                else:
                    current_state = gameController.get_game_state()
                    player_id = 2 if player[-1] == 2 else -1
                    state_with_player = torch.tensor([player_id] + current_state).float()
                    softmax_distr = model.forward(state_with_player).detach().numpy()
                    softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)

                    action = Utils.make_max_move_from_distribution(softmax_distr_re_normalized, size)
                    print("Machine picks move", action)
                    gameController.make_move(action, player)

                if gameController.game_is_won():
                    if player == 'Player1':
                        print('Human wins!')
                    else:
                        print("Computer wins")
                    break

