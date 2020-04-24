import os
from itertools import combinations, cycle
from Config import *
from Actor import Actor
from Hex import Hex
from GameController import GameController
import Utils
from scipy.special import softmax
import random

class Tournament:
    def __init__(self, board_size, games_to_play, models):
        self.board_size_in_tournament = board_size
        self.games_to_play = games_to_play
        self.models = models

    @staticmethod
    def load_contesters(board_size):
        models = {}
        for filename in sorted(os.listdir('./models/boardsize_'+str(board_size))):
            #episode = int(filename.split("_")[-1].split(".")[0])
            # if episode in []:
            model = Actor.load_model('./models/boardsize_'+str(board_size) +'/' + filename,
                                    board_size,
                                      HIDDEN_LAYERS,
                                      LEARNING_RATE,
                                      ACTIVATION,
                                      OPTIMIZER)
            models[filename] = model
            print(filename)

        return models

    @staticmethod
    def load_consters_for_boardsize_5_demo(board_size=5):
        models = {}
        for filename in sorted(os.listdir('./models/boardsize_'+str(board_size)+'_demo')):
            model = Actor.load_model('./models/boardsize_'+str(board_size) +'_demo/' + filename,
                                    board_size,
                                      HIDDEN_LAYERS,
                                      LEARNING_RATE,
                                      ACTIVATION,
                                      OPTIMIZER)
            models[filename] = model
            print(filename)

        return models

    def run_tournament(self):
        game = Hex(size=BOARD_SIZE_IN_TOURNAMENT)
        gameController = GameController(game, visualize=False)

        matchup_combinations = list(combinations(self.models.keys(), 2))

        scores = {}
        for key in self.models.keys():
            scores[key] = 0

        for matchup in matchup_combinations:  # player1 = matchup[0], player2 = matchup[1]
            for game in range(1, M_GAMES_TO_PLAY_IN_TOPP + 1):
                gameController.reset_game()
                GAME_CYCLE = cycle(['Player1', 'Player2']) if game % 2 == 0 else cycle(['Player2', 'Player1'])

                for player in GAME_CYCLE:
                    current_state = gameController.get_game_state()
                    player_id = 1 if player[-1] == '1' else -1
                    state_repr_for_net = gameController.get_state_repr_of_game_for_net(current_state, player_id,
                                                                                       self.board_size_in_tournament)
                    net_to_use = self.models[matchup[0]] if player == 'Player1' else self.models[matchup[1]]

                    softmax_distr = softmax(net_to_use.forward(state_repr_for_net).detach().numpy())
                    softmax_distr_re_normalized = (Utils.re_normalize(current_state, softmax_distr))

                    if random.uniform(0, 1) < 0.25:
                        action = Utils.move_from_distribution(softmax_distr_re_normalized, self.board_size_in_tournament)
                        gameController.make_move(action, player)
                    else:
                        action = Utils.max_move_from_distribution(softmax_distr_re_normalized, self.board_size_in_tournament)
                        gameController.make_move(action, player)

                    if gameController.game_is_won():
                        if player == 'Player1':
                            scores[matchup[0]] += 1
                        else:
                            scores[matchup[1]] += 1
                        break

        print("\nFinal scores after", M_GAMES_TO_PLAY_IN_TOPP, "games against one another, total",
              sum(scores.values()), "games played between the", len(scores.keys()), "players")
        print("Format: (Number of wins in tournament, model name)")
        print(sorted(((v, k) for k, v in scores.items()), reverse=True))


if __name__ == '__main__':
    BOARD_SIZE_IN_TOURNAMENT = 5
    #models = Tournament.load_contesters(BOARD_SIZE_IN_TOURNAMENT)
    models = Tournament.load_consters_for_boardsize_5_demo()

    topp = Tournament(BOARD_SIZE_IN_TOURNAMENT, M_GAMES_TO_PLAY_IN_TOPP, models)
    topp.run_tournament()

