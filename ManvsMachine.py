import os
from Config import *
from Actor import Actor
import Utils
from Hex import Hex
from GameController import GameController
from itertools import cycle
from scipy.special import softmax
import random


def load_model(model_path, size, hidden_layers):
    actor = Actor(size, hidden_layers, LEARNING_RATE, ACTIVATION, OPTIMIZER)
    model = actor.net
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


if __name__ == '__main__':
    size = 4
    hidden_layers = [48, 48]
    game = Hex(size=size)
    model = load_model('./models/boardsize_'+str(size) +'/net_after_episode_300.pt', size, hidden_layers)
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
                    action = None
                    while action not in gameController.get_all_valid_moves():
                        try:
                            action = input("What is your action human? e.g. type 0 0 ")
                            action = tuple(map(int, action.split(' ')))
                        except ValueError:
                            continue
                    gameController.make_move(action, player)
                else:
                    current_state = gameController.get_game_state()
                    player_id = 1 if player[-1] == '1' else -1
                    number_of_cells = size * size
                    number_of_taken_cells = gameController.get_number_of_pieces_on_board()
                    number_of_free_cells = number_of_cells - number_of_taken_cells
                    frac_free = number_of_free_cells / number_of_cells
                    frac_taken = number_of_taken_cells / number_of_cells
                    player_1_taken_endwall1, player_1_taken_endwall2 = gameController.get_outer_walls_set_for_player(1)
                    player_2_taken_endwall1, player_2_taken_endwall2 = gameController.get_outer_walls_set_for_player(2)

                    feat_eng_state = [frac_free, frac_taken, player_1_taken_endwall1, player_1_taken_endwall2,
                                      player_2_taken_endwall1, player_2_taken_endwall2]

                    state_for_net = torch.tensor([player_id] + current_state + feat_eng_state).float()

                    softmax_distr = softmax(model.forward(state_for_net).detach().numpy())
                    softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)
                    #print()
                    if random.uniform(0, 1) < 0:
                        action = Utils.make_move_from_distribution(softmax_distr_re_normalized, size, verbose = True)
                        gameController.make_move(action, player)
                    else:
                        action = Utils.make_max_move_from_distribution(softmax_distr_re_normalized, size, verbose=True)
                        gameController.make_move(action, player)

                if gameController.game_is_won():
                    if player == 'Player1':
                        print('Human wins!')
                    else:
                        print("Computer wins")
                    break

