import random


class StateManager:
    def __init__(self, game):
        self.game = game

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
        if random_move is None:
            return
        else:
            self.game.make_move(random_move, player)
