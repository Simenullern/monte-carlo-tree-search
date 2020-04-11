from BoardCell import BoardCell
from Stack import Stack
import copy
import networkx as nx
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class Hex:
    def __init__(self, size):
        assert(3 <= size <= 10)
        self.size = size
        self.cells = [[-1 for i in range(size)] for j in range(size)]
        self.potential_neighbors = [(-1, 1), (-1, 0), (0, -1), (0, 1), (1, 0), (1, -1)]
        self.init()

    def get_size(self):
        return self.size

    def get_state(self):
        return self.cells

    def get_succ_state(self, action):
        cop = copy.deepcopy(self)
        cop.verbose = False
        cop.make_move(action, "_")
        return cop.get_state()

    def init(self, verbose = False):
        for i in range(0, len(self.cells)):
            row = self.cells[i]
            for j in range(0, len(row)):
                row[j] = BoardCell(loc=(i,j))
        self.register_neighbors()

    def register_neighbors(self):
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                for pn in self.potential_neighbors:
                    try:
                        if r+pn[0] < 0 or c + pn[1] < 0:
                            continue  # Don't allow for negative indexes
                        neighbor = self.cells[r+pn[0]][c+pn[1]]
                        current_cell.add_neighbor(neighbor)
                    except IndexError:
                        continue

    def is_won(self):
        # Player1 (red) wins if all rows have been spanned in a chain of pieces
        # Player2 (black) wins if all cols have been spanned in a chain of pieces

        # Check for red with a depth first search
        visited = set()
        stack = Stack()
        for c in range(0, len(self.cells[0])):
            current_cell = self.cells[0][c]
            if current_cell.get_fill() == 'red' and current_cell not in visited:
                stack.push(current_cell)
                while not stack.is_empty():
                    cell = stack.pop()
                    if cell.get_loc()[0] == self.size-1:
                        #print('RED WON')
                        return True
                    for neighbor in cell.get_neighbors():
                        if neighbor.get_fill() == 'red' and neighbor not in visited:
                            stack.push(neighbor)
                            visited.add(neighbor)

        # Check for black with a depth first search
        visited = set()
        stack = Stack()
        for r in range(0, len(self.cells[0])):
            current_cell = self.cells[r][0]
            if current_cell.get_fill() == 'black' and current_cell not in visited:
                stack.push(current_cell)
                while not stack.is_empty():
                    cell = stack.pop()
                    if cell.get_loc()[1] == self.size - 1:
                        #print('black WON')
                        return True
                    for neighbor in cell.get_neighbors():
                        if neighbor.get_fill() == 'black' and neighbor not in visited:
                            stack.push(neighbor)
                            visited.add(neighbor)

        return False

    def get_all_valid_moves(self):
        possible_moves = []
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                current_cell_loc = current_cell.get_loc()
                if not current_cell.is_filled():
                    possible_moves.append(current_cell_loc)

        return possible_moves

    def make_move(self, move, player):
        # on the form a where a is the loc to place
        for valid_move in self.get_all_valid_moves():
            if move == valid_move:
                row_to = move[0]
                col_to = move[1]
                if player == 'Player1':
                    self.cells[row_to][col_to].set_fill('red')
                else:
                    self.cells[row_to][col_to].set_fill('black')
                return
        raise Exception(str(move) + ' is not a valid move')

    def get_hashable_state(self):
        string = ""
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                if current_cell.is_filled():
                    string += "1"
                else:
                    string += "0"
        return string

    def visualize(self):
        G = nx.Graph(tight_layout = False)

        # Add Nodes
        node_colors = []
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                if current_cell.is_filled():
                    node_colors.append(current_cell.get_fill())
                else:
                    node_colors.append('lightgray')
                G.add_node((r, c))

        edge_colors = []
        cache = set()
        # Add Edges
        for r in range(0, len(self.cells)):
            for c in range(0, len(self.cells[r])):
                current_cell = self.cells[r][c]
                current_loc = current_cell.get_loc()
                for neighbor in current_cell.get_neighbors():
                    neighbor_loc = neighbor.get_loc()
                    if not (current_loc, neighbor_loc) in cache:
                        G.add_edge(current_loc, neighbor_loc)

                        if current_loc[0] == 0 and neighbor.get_loc()[0] == 0:
                            edge_colors.append('red')
                        elif current_loc[0] == self.size-1 and neighbor.get_loc()[0] == self.size-1:
                            edge_colors.append('red')
                        elif current_loc[1] == 0 and neighbor.get_loc()[1] == 0:
                            edge_colors.append('black')
                        elif current_loc[1] == self.size - 1 and neighbor.get_loc()[1] == self.size - 1:
                            edge_colors.append('black')
                        else:
                            edge_colors.append('gray')

                    cache.update([(current_loc, neighbor_loc), (neighbor_loc, current_loc)])

        # Draw the board
        options = {
            'node_size': 300,
            'font_size': 8,
            'font_color': 'black',
            'node_color': node_colors,
            'edge_color': edge_colors,
            'pos': nx.spring_layout(G, seed=5),

        }

        nx.draw(G, with_labels=True, **options)
        plt.show()


if __name__ == '__main__':
    Board = Hex(size=4)
    Board.visualize()
    print(Board.get_all_possible_moves())
    Board.make_move((1, 1), 'Player1')
    print(Board.is_won())
    Board.make_move((3, 0), 'Player2')
    print(Board.is_won())
    Board.make_move((2, 1), 'Player1')
    print(Board.is_won())
    Board.make_move((2, 2), 'Player2')
    print(Board.is_won())
    Board.make_move((0, 1), 'Player1')
    print(Board.is_won())
    Board.make_move((2, 3), 'Player2')
    print(Board.is_won())
    Board.make_move((0, 0), 'Player1')
    print(Board.is_won())
    Board.make_move((3, 1), 'Player2')
    print(Board.is_won())
    Board.visualize()
