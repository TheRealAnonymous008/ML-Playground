
PLAYER_ONE = 1 
PLAYER_TWO = -1
EMPTY = 0
WILDCARD_BOX = -1

# Expected 1 <= row, col <= 9 
def get_box(row, column):
    row = row - 1
    column = column - 1

    r = row // 3
    c = column // 3 

    return 3 * r + c

def get_box_coords(row, column):
    row = row - 1
    column = column - 1

    r = row % 3
    c = column % 3

    return 3 * r + c

def get_micro_winner(box):
    for i in range(0, 3):
        # Check rows
        if box[3 * i] == box[3 * i + 1] and box[3 * i] == box[3 * i + 2] and box[3 * i] != EMPTY:
            return box[3 * i]
        
        # Check columns
        if box[i] == box[i + 3] and box[i] == box[i + 6] and box[i] != EMPTY:
            return box[i]
        
    # Check Diagonals
    if box[0] == box[4] and box[0] == box[8] and box[0] != EMPTY:
        return box[0]
    if box[2] == box[4] and box[4] == box[6] and box[2] != EMPTY:
        return box[2] 
    
    return EMPTY

CENTER_COORDS = [(1, 1), (1, 4), (1, 7), (4, 1), (4, 4), (4, 7), (7, 1), (7, 4), (7, 7)]
class GameState: 
    # Board state is arranged as (indices)
    # 0 1 2 
    # 3 4 5
    # 6 7 8    
    def __init__(self, board = None, player = PLAYER_ONE):        
        if board is None:
            self.board = [[EMPTY for i in range(0, 9)] for i in range(0, 9)]
        else:  
            if len(board) != 9 or len(board[0]) != 9:
                raise ValueError("Inappropriate argument value")
            self.board = board 

        self.current_player = player
        self.current_box = WILDCARD_BOX

    def get_big_board(self):
        big_board = []

        for c in range(0, 9):
            box = self.get_box_at(c)
            big_board.append(get_micro_winner(box))

        return big_board
    
    def get_box_at(self, coord):
        box = []
        row, column = CENTER_COORDS[coord]
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                rb = row + i 
                cb = column + j
                box.append(self.board[rb][cb])

        return box


    def get_current_box(self):
        if self.current_box != WILDCARD_BOX:
            return self.get_box_at(self.current_box)
        return None
    
    def get_big_winner(self):
        return get_micro_winner(self.get_big_board())
    
    def get_successors(self):
        succ : list[GameState] = []

        if self.get_big_winner() != EMPTY:
            return None

        bboard = self.get_big_board()

        for i in range(0, 9):
            if self.current_box != WILDCARD_BOX and i != self.current_box:
                continue 

            if bboard[i] != EMPTY: 
                continue 
            else:
                bx, by = CENTER_COORDS[i]
                for x in [-1, 0, 1]:
                    for y in [-1, 0, 1]:
                        cx = bx + x 
                        cy = by+ y 
                        if self.board[cx][cy] != EMPTY:
                            continue 
                        else: 
                            g = GameState()
                            g.board[cx][cy] = self.current_player
                            succ.append(g)

        if len(succ) == 0:
            return None 
        return succ


class Game: 
    def __init__(self, state = None):
        if state is None:
            self.state = GameState()
        else:
            self.state = state

    def play(self, row, column):
        if row < 1 or row > 9 or column < 1 or column > 9:
            print("Invalid Range. Try again")
            return 
        
        if get_box(row, column) != self.state.current_box and self.state.current_box != WILDCARD_BOX:
            print("Invalid Box. Try again")
            return 

        self.state.board[row - 1][column - 1] = self.state.current_player

        self.switch_box(row, column)
        self.switch_player()
    
    def switch_box(self, row, column):
        self.state.current_box = get_box_coords(row, column)
        if get_micro_winner(self.state.get_current_box()) != EMPTY:
            self.state.current_box = WILDCARD_BOX

    def switch_player(self):
        if self.state.current_player == PLAYER_ONE:
            self.state.current_player = PLAYER_TWO
        else: 
            self.state.current_player = PLAYER_ONE

    def get_current_state(self):
        return self.state
    
    def get_current_player(self):
        return self.state.current_player 
    
    def get_winner(self):
        winner = self.state.get_big_winner()
        return winner

    def print_board(self):
        for i in range(0, 9):
            s = "|"
            for j in range(0, 9):
                c = "- -" 
                if self.state.board[i][j] == PLAYER_ONE:
                    c = "-X-"
                elif self.state.board[i][j] == PLAYER_TWO:
                    c = "-O-"

                if j % 3 == 2: 
                    c += "|"
                
                s += c 
            
            print(s)
            if (i % 3 == 2):
                print()

    def print_big_board(self):
        bb = self.state.get_big_board()

        print("| " + str(bb[0]) + " | " + str(bb[1]) + " | " + str(bb[2]) + " |") 
        print("| " + str(bb[3]) + " | " + str(bb[4]) + " | " + str(bb[5]) + " |")            
        print("| " + str(bb[6]) + " | " + str(bb[7]) + " | " + str(bb[8]) + " |")            

    def print_winner(self):
        winner = self.get_winner()
        if winner == EMPTY:
            print("No Winner")
        elif winner == PLAYER_ONE:
            print("X Wins")
        elif winner == PLAYER_TWO:
            print("O Wins")