
 
EMPTY = 0 

def get_legal_moves (board):
    
    moves = []

    for col in range(len(board[0])):

        last_open_row = len(board)

        for row in range(len(board) - 1, -1, -1):
            if board[row][col] != EMPTY:
                last_open_row = row

        if last_open_row > 0:
            moves.append((last_open_row - 1, col))
    
    return moves

def reset_board (height=6, width=7):
    return [[EMPTY for i in range(width)] for j in range(height)]

def check_win (board, win_row):
    has_empty = False
    board_width = len(board[0])
    board_height = len(board)
    for i in range(board_height):
        for j in range(board_width):
            if board[i][j] == EMPTY:
                has_empty = True
            
            if j <= board_width - win_row:
                sum_ = 0
                for k in range(win_row):
                    sum_ += board[i][j + k]
                if abs(sum_) == win_row:
                    return board[i][j]
            
            if i <= board_height - win_row:
                sum_ = 0
                for k in range(win_row):
                    sum_ += board[i + k][j]
                if abs(sum_) == win_row:
                    return board[i][j]
            
            if i <= board_height - win_row and \
                j <= board_width - win_row:
                sum_ = 0
                for k in range(win_row):
                    sum_ += board[i + k][j + k]
                if abs(sum_) == win_row:
                    return board[i][j]
            
            if i <= board_height - win_row and \
                j >= win_row - 1:
                sum_ = 0
                for k in range(win_row):
                    sum_ += board[i + k][j - k]
                if abs(sum_) == win_row:
                    return board[i][j]

    if not has_empty:
        return 2

    return 0        
    
    
 