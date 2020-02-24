from Stack import Stack


class Nim:
    def __init__(self, no_of_starting_stones, max_num_of_stones_to_take, verbose=False):
        assert(no_of_starting_stones > 100 and max_num_of_stones_to_take > 1)
        self.no_of_starting_stones = no_of_starting_stones
        self.max_num_of_stones_to_take = max_num_of_stones_to_take
        self.verbose = verbose
        self.pile = Stack()
        self.init_pile()

    def init_pile(self):
        for stone in range(0, self.no_of_starting_stones):
            self.pile.push()
        if self.verbose:
            print("Start pile:", self.no_of_starting_stones, "stones")

    def make_move(self, num_of_stones_to_pick, player):
        if num_of_stones_to_pick > self.max_num_of_stones_to_take:
            raise ValueError(player, "cant pick", num_of_stones_to_pick, "only allowed", self.max_num_of_stones_to_take)

        if self.pile.get_len() - num_of_stones_to_pick < 0:
            raise ValueError(player, "cant pick", num_of_stones_to_pick, "when only", self.pile.get_len(), "remains")

        assert num_of_stones_to_pick in self.get_all_valid_moves() ###### dup from abov

        for stone in range(0, len(self.max_num_of_stones_to_take)):
            self.pile.pop()

        if self.verbose:
            print(player, "selects", num_of_stones_to_pick, "stones: Remaining stones =", self.pile.get_len())
            if self.is_won():
                print(player, "wins")

    def get_all_valid_moves(self):
        valid_number_of_stones_to_pick = []
        number_of_stones_remain = self.pile.get_len()
        if number_of_stones_remain > 0:
            pick = 1
            while number_of_stones_remain - pick > -1 and pick <= self.max_num_of_stones_to_take:
                valid_number_of_stones_to_pick.append(pick)
                pick += 1
            return valid_number_of_stones_to_pick
        else:
            return []

    def is_won(self):
        return self.pile.get_len() == 0


if __name__ == '__main__':
    Nim = Nim(101, 8, True)
    print(Nim.get_all_valid_moves())