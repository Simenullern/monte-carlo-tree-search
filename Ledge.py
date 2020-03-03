import random
import copy


class Ledge:
    def __init__(self, init_board, verbose):
        assert(1 < len(init_board) <= 20 and -1 < init_board.count(1) < len(init_board) and init_board.count(2) == 1)
        self.verbose = verbose

        self.init_board = init_board
        self.board = None
        self.init()

    def get_state(self):
        return self.board

    def get_succ_state(self, action):
        cop = copy.deepcopy(self)
        cop.verbose = False
        cop.make_move(action, "_")
        return cop.get_state()

    def init(self):
        self.board = copy.copy(self.init_board)
        if self.verbose:
            print("Start board", self.board)

    def make_move(self, move, player):
        assert move in self.get_all_valid_moves()
        brick_type = move[0]
        index_from = move[1]
        index_to = move[2]

        self.board[index_to] = self.board[index_from]
        self.board[index_from] = 0

        if index_to == -1 and brick_type == 'gold':
            self.board[index_to] = 0
            if self.verbose:
                print(player, "pick up gold and wins:", self.board)
        elif index_to == -1 and brick_type == 'copper':
            self.board[index_to] = 0
            if self.verbose:
                print(player, "picks up copper:", self.board)
        else:
            if self.verbose:
                print(player, "moves", brick_type, "from cell", index_from, "to", index_to, ":", self.board)

    def get_all_valid_moves(self):
        valid_moves = []
        # Returns a list of tuples (a, b, c) where
        # a = 'copper' or 'gold', b = index of cell moving from, c = index of cell moving to
        a, b, c = None, None, None
        for forward_index in range(0, len(self.board)):
            if self.board[forward_index] == 0:
                continue

            elif self.board[forward_index] == 2:
                a = 'gold'
                if forward_index == 0:
                    b = forward_index
                    c = -1
                    valid_moves.append((a, b, c))

            elif self.board[forward_index] == 1:
                a = 'copper'
                if forward_index == 0:
                    b = forward_index
                    c = -1
                    valid_moves.append((a, b, c))

            for backward_index in range(forward_index-1, -1, -1):
                if self.board[backward_index] == 0:
                    b = forward_index
                    c = backward_index
                    valid_moves.append((a, b, c))
                else:
                    break

        return valid_moves

    def is_won(self):
        return 2 not in self.board


if __name__ == '__main__':
    Ledge = Ledge(10, 3, True)
    print(Ledge.get_all_valid_moves())