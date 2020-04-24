import math
from BasicClientActorAbs import BasicClientActorAbs
import torch
from Actor import Actor
from Config import *
import Utils
import random
from scipy.special import softmax


class BasicClientActor(BasicClientActorAbs):

    def __init__(self, IP_address=None, verbose=True):
        self.series_id = -1
        BasicClientActorAbs.__init__(self, IP_address, verbose=verbose)

    def handle_get_action(self, oht_state):
        """
        Here you will use the neural net that you trained using MCTS to select a move for your actor on the current board.
        Remember to use the correct player_number for YOUR actor! The default action is to select a random empty cell
        on the board. This should be modified.
        :param state: The current board in the form (1 or 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), where
        1 or 2 indicates the number of the current player.  If you are player 2 in the current series, for example,
        then you will see a 2 here throughout the entire series, whereas player 1 will see a 1.
        :return: Your actor's selected action as a tuple (row, column)
        """

        BOARD_SIZE = 6
        HIDDEN_LAYERS = [96, 96]
        next_move = None

        current_state, state_for_net = BasicClientActor.map_oht_format_to_my_format(oht_state, BOARD_SIZE)

        net_to_use = Actor.load_model('./models/boardsize_'+str(BOARD_SIZE) +'/net_after_episode_4550.pt',
                                      BOARD_SIZE,
                                      HIDDEN_LAYERS,
                                      LEARNING_RATE,
                                      ACTIVATION,
                                      OPTIMIZER)

        softmax_distr = softmax(net_to_use.forward(state_for_net).detach().numpy())
        softmax_distr_re_normalized = (Utils.re_normalize(current_state, softmax_distr))

        if random.uniform(0, 1) < 0:
            next_move = Utils.move_from_distribution(softmax_distr_re_normalized, BOARD_SIZE, verbose=False)
        else:
            next_move = Utils.max_move_from_distribution(softmax_distr_re_normalized, BOARD_SIZE, verbose=False)

        return next_move

    def handle_series_start(self, unique_id, series_id, player_map, num_games, game_params):
        """
        Set the player_number of our actor, so that we can tell our MCTS which actor we are.
        :param unique_id - integer identifier for the player within the whole tournament database
        :param series_id - (1 or 2) indicating which player this will be for the ENTIRE series
        :param player_map - a list of tuples: (unique-id series-id) for all players in a series
        :param num_games - number of games to be played in the series
        :param game_params - important game parameters.  For Hex = list with one item = board size (e.g. 5)
        :return

        """
        self.series_id = series_id
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_start(self, start_player):
        """
        :param start_player: The starting player number (1 or 2) for this particular game.
        :return
        """
        self.starting_player = start_player
        #############################
        #
        #
        # YOUR CODE (if you have anything else) HERE
        #
        #
        ##############################

    def handle_game_over(self, winner, end_state):
        """
        Here you can decide how to handle what happens when a game finishes. The default action is to print the winner and
        the end state.
        :param winner: Winner ID (1 or 2)
        :param end_state: Final state of the board.
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        ##############################
        print("Game over, these are the stats:")
        print('Winner: ' + str(winner))
        print('End state: ' + str(end_state))

    def handle_series_over(self, stats):
        """
        Here you can handle the series end in any way you want; the initial handling just prints the stats.
        :param stats: The actor statistics for a series = list of tuples [(unique_id, series_id, wins, losses)...]
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Series ended, these are the stats:")
        print(str(stats))

    def handle_tournament_over(self, score):
        """
        Here you can decide to do something when a tournament ends. The default action is to print the received score.
        :param score: The actor score for the tournament
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("Tournament over. Your score was: " + str(score))

    def handle_illegal_action(self, state, illegal_action):
        """
        Here you can handle what happens if you get an illegal action message. The default is to print the state and the
        illegal action.
        :param state: The state
        :param action: The illegal action
        :return:
        """
        #############################
        #
        #
        # YOUR CODE HERE
        #
        #
        #############################
        print("An illegal action was attempted:")
        print('State: ' + str(state))
        print('Action: ' + str(illegal_action))

    @staticmethod
    def map_oht_format_to_my_format(oht_state, board_size):
        player_id = 1 if oht_state[0] == 1 else -1
        current_state = []
        pieces_on_board = 0

        player_1_taken_endwall1 = 0
        player_1_taken_endwall2 = 0
        player_2_taken_endwall1 = 0
        player_2_taken_endwall2 = 0

        for i in range(1, len(oht_state)):
            piece = oht_state[i]
            if not piece == 0:
                pieces_on_board += 1

                if i <= board_size and piece == 1:
                    player_1_taken_endwall1 = 1

                if i > (board_size - 1) * board_size and piece == 1:
                    player_1_taken_endwall2 = 1

                if i % board_size == 1 and piece == 2:
                    player_2_taken_endwall1 = 1

                if i % board_size == 0 and piece == 2:
                    player_2_taken_endwall2 = 1

            if piece == 0 or piece == 1:
                current_state.append(piece)
            elif piece == 2:
                current_state.append(-1)

        number_of_cells = board_size * board_size
        number_of_taken_cells = pieces_on_board
        number_of_free_cells = number_of_cells - number_of_taken_cells
        frac_free = number_of_free_cells / number_of_cells
        frac_taken = number_of_taken_cells / number_of_cells

        feat_eng_state = [frac_free, frac_taken, player_1_taken_endwall1, player_1_taken_endwall2,
                          player_2_taken_endwall1, player_2_taken_endwall2]

        state_for_net = torch.tensor([player_id] + current_state + feat_eng_state).float()

        return current_state, state_for_net


if __name__ == '__main__':
    bsa = BasicClientActor(verbose=False)
    bsa.connect_to_server()
