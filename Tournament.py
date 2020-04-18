import os
from itertools import combinations, cycle
import torch
from Config import *
from Actor import Actor
from Hex import Hex
from GameController import GameController
import Utils
from scipy.special import softmax
import random


def load_contesters(board_size, layers):
    models = {}

    for filename in sorted(os.listdir('./models/boardsize_'+str(board_size))):
        actor = Actor(board_size, layers, LEARNING_RATE, ACTIVATION, OPTIMIZER)
        model = actor.net
        model.load_state_dict(torch.load('./models/boardsize_'+str(board_size)+'/'+filename))
        model.eval()
        models[filename] = model
        print(filename)

        #current_state = [-1, 1, 1, 0, 1, -1, 0, -1, 1]
        #player_id = -1
        #feat_eng_state = [2/9, 7/9, 1, 1, 1, 1]
        #state_for_net = torch.tensor([player_id] + current_state + feat_eng_state).float()
        #state_with_player = torch.tensor([1, 0, 1, 0, 0, 0, 1, 0, 0, -1, 1, -1, 0, -1, 0, 0, 0]).float()
        #softmax_distr = softmax(model.forward(state_for_net).detach().numpy())
        #print(softmax_distr)
        #softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)
        #print(softmax_distr_re_normalized, '\n')

    breakpoint()

    #breakpoint()
    return models


if __name__ == '__main__':
    size = 5
    hidden_layers = [48, 48]
    models = load_contesters(size, hidden_layers)
    matchup_combinations = list(combinations(models.keys(), 2))

    game = Hex(size=size)
    gameController = GameController(game, visualize=False)
    start_state = gameController.get_game_state()

    scores = {}
    for key in models.keys():
        scores[key] = 0

    for matchup in matchup_combinations:
        player1 = matchup[0]
        player2 = matchup[1]

        for game in range(1, M_GAMES_TO_PLAY_IN_TOPP+1):
            gameController.reset_game()
            GAME_CYCLE = cycle(['Player1', 'Player2']) if game % 2 == 0 else cycle(['Player2', 'Player1'])

            for player in GAME_CYCLE:
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

                net_to_use = models[matchup[0]] if player == 'Player1' else models[matchup[1]]

                softmax_distr = softmax(net_to_use.forward(state_for_net).detach().numpy())
                softmax_distr_re_normalized = (Utils.re_normalize(current_state, softmax_distr))
                if random.uniform(0, 5) < 1:
                    action = Utils.make_move_from_distribution(softmax_distr_re_normalized, size)
                    gameController.make_move(action, player)
                else:
                    action = Utils.make_max_move_from_distribution(softmax_distr_re_normalized, size)
                    gameController.make_move(action, player)

                if gameController.game_is_won():
                    #print(player1, "is meeting", player2, "for game", game, ":", player, "wins!")
                    #if game % 2 == 0:
                        #print('\tPlayer1 started')
                    #else:
                        #print("\tPlayer2 started")

                    if player == 'Player1':
                        scores[matchup[0]] += 1
                    else:
                        scores[matchup[1]] += 1
                    break

    print("Final scores after", M_GAMES_TO_PLAY_IN_TOPP, "games against one another, total",
          sum(scores.values()), "games played between the", len(scores.keys()), "players")
    print("Format: Number of wins in tournament, model name")
    print(sorted(((v,k) for k,v in scores.items()), reverse=True))
