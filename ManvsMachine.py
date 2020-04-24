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
    BOARD_SIZE = 6
    hidden_layers = [96, 96]
    game = Hex(size=BOARD_SIZE)
    model = load_model('./models/boardsize_'+str(BOARD_SIZE) +'/net_after_episode_4550.pt', BOARD_SIZE, hidden_layers)
    gameController = GameController(game, visualize=True)

    while True:
        for game in range(1, 11):
            gameController.reset_game()
            gameController.visualize()

            GAME_CYCLE = cycle(['Player1', 'Player2']) if game % 2 == 0 else cycle(['Player2', 'Player1'])
            # Player 1 is YOU, the human

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
                    state_repr_for_net = gameController.get_state_repr_of_game_for_net(current_state, player_id, BOARD_SIZE)

                    softmax_distr = softmax(model.forward(state_repr_for_net).detach().numpy())
                    softmax_distr_re_normalized = Utils.re_normalize(current_state, softmax_distr)

                    if random.uniform(0, 1) < 0:
                        action = Utils.move_from_distribution(softmax_distr_re_normalized, BOARD_SIZE, verbose = True)
                        gameController.make_move(action, player)
                    else:
                        action = Utils.max_move_from_distribution(softmax_distr_re_normalized, BOARD_SIZE, verbose=True)
                        gameController.make_move(action, player)

                if gameController.game_is_won():
                    if player == 'Player1':
                        print('Human wins!')
                    else:
                        print("Computer wins")
                    break

