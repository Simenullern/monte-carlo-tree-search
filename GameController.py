import random
import copy


class GameController:
    def __init__(self, game):
        self.game = game
        self.games_won = {'Player1': 0, 'Player2': 0}

    def summarize_stats(self):
        num_of_games = self.games_won['Player1'] + self.games_won['Player2']
        p1_percentage = round((self.games_won['Player1'] / num_of_games) * 100, 2)
        print("\t Player1 wins " + str(self.games_won['Player1']) + str(" of ") + str(num_of_games) +\
            " games (" + str(p1_percentage) + "%)")

    def get_game_state(self):
        return copy.deepcopy(self.game.get_state())

    def get_succ_state(self, action):
        this = copy.deepcopy(self)
        this.make_move(action, "_")
        return this.get_game_state()

    def get_copy_for_simulation(self, disable_verbosity=True):
        this = copy.deepcopy(self)
        if disable_verbosity:
            this.disable_verbosity()
        return this

    def disable_verbosity(self):
        self.game.verbose = False

    def register_victory(self, player):
        self.games_won[player] += 1

    def reset_game(self):
        self.game.init()

    def game_is_won(self):
        return self.game.is_won()

    def game_is_on(self):
        return not self.game_is_won()

    def get_all_valid_moves(self):
        return self.game.get_all_valid_moves()

    def get_random_move(self):
        if self.game_is_on():
            available_moves = self.get_all_valid_moves()
            random.shuffle(available_moves)
            return available_moves[0]
        else:
            return None

    def make_random_move(self, player):
        random_move = self.get_random_move()
        self.game.make_move(random_move, player)
        return random_move

    def make_move(self, move, player):
        self.game.make_move(move, player)
        return move
