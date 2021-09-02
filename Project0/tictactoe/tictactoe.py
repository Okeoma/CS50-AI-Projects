"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # X makes the first move if the board is empty

    """
    if board == initial_state():
       return X
    else:
    """
    # Initializes count for X and O moves
    count_x = 0
    count_o = 0
    if board == initial_state():
        return X

    # Increments count for X and O
    for i in board:
        for j in i:
            if j == X:
                count_x += 1
            if j == O:
                count_o += 1

    # Checks the next player's turn based on no. of moves
    if count_x <= count_o:
        return X
    else:
        return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Confirms the rows and columns that are empty to check if actions are permissible on them
    permissible_actions = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                permissible_actions.add((i, j))

    return permissible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Error handling when a wrong move is made
    permissible_actions = actions(board)
    if action not in permissible_actions:
        raise Exception("The move is not allowed")

    current_board = copy.deepcopy(board)
    current_board[action[0]][action[1]] = player(current_board)

    return current_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Assess board for player X and player 0
    for target in [X, O]:

        # Confirm if any row is complete for a player
        for i in range(3):
            if all(board[i][j] == target for j in range(3)):
                return target

        # Confirm if any column is complete for a player
        for j in range(3):
            if all(board[i][j] == target for i in range(3)):
                return target

        # Confirm if any crosswise pattern is complete for a player
        crosswise = [[(0, 0), (1, 1), (2, 2)], [(2, 0), (1, 1), (0, 2)]]
        for cross in crosswise:
            if all(board[i][j] == target for (i, j) in cross):
                return target
      
    # When there is no winner
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Terminate game if there is no more move(tie) or if there is a winner
    if not actions(board) or winner(board) is not None:
        return True
    else:  # If game is still on
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    # i and j represents the first action to be made on the board
    i = 0
    j = 1
    if board == initial_state():
        return i, j

    if player(board) == X:
        v = -math.inf

    if player(board) == O:
        v = math.inf

    for action in actions(board):

        if player(board) == X:
            new_v = max(v, min_value(result(board, action)))

        if player(board) == O:
            new_v = min(v, max_value(result(board, action)))

        if new_v != v:
            v = new_v
            optimal_action = action

    return optimal_action


def max_value(board):
    """
    Returns the maximum utility of the current board.
    """

    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    """
    Returns the minimum utility of the current board.
    """

    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
