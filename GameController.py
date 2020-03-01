import random


class GameController:
    def __init__(self, game):
        self.game = game
        self.games_won = {'Player1': 0, 'Player2': 0}

    def summarize_stats(self):
        num_of_games = self.games_won['Player1'] + self.games_won['Player2']
        p1_percentage = round((self.games_won['Player1'] / num_of_games) * 100, 2)
        print("\t Player1 wins " + str(self.games_won['Player1']) + str(" of ") + str(num_of_games) +\
            " games (" + str(p1_percentage) + "%)")

    def register_victory(self, player):
        self.games_won[player] += 1

    def reset_game(self):
        self.game.init()

    def game_is_won(self):
        return self.game.is_won()

    def game_is_on(self):
        return not self.game_is_won()

    def get_random_move(self):
        if self.game_is_on():
            available_moves = self.game.get_all_valid_moves()
            random.shuffle(available_moves)
            return available_moves[0]
        else:
            return None

    def make_random_move(self, player):
        random_move = self.get_random_move()
        self.game.make_move(random_move, player)
        return random_move
