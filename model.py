import random
import json
import copy 
import os
import numpy as np
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
# import keras
import tensorflow as tf
from tensorflow import keras

import game

FIRST_PLAYER = 1
SECOND_PLAYER = -1

class NN:
    
    def __init__(self, path=None):

        inputs = x = keras.Input(shape=(8, 8, 2))
        x = keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.Conv2D(filters=32, kernel_size=6, padding="same")(x)
        x = keras.layers.Activation("relu")(x)                
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(320, activation="relu")(x)
        outputs = keras.layers.Dense(1, activation="tanh")(x)

        self.model = keras.Model(inputs, outputs, name="connect4_model") 

        if path != None:
            self.model.load_weights(path)

        self.model.summary()
        self.model.compile(
            loss="mean_squared_error",
            optimizer=keras.optimizers.Adam())

    def predict (self, x):
        return self.model.predict(np.array(x).reshape(1, 8, 8, 2))

    def train (self, x, y, epochs):
        return self.model.train_on_batch(x, y)

    def fit (self, x, y, epochs):
        return self.model.fit(x, y, epochs=epochs)
        
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def search (nn, board, color):
    legal_moves = game.get_legal_moves(board)
    possible_moves = np.zeros((7,))
    
    for move in legal_moves:
        board[move[0]][move[1]] = color
        pred = nn.predict(get_board_by_player(board, color))
        possible_moves[move[1]] = pred[0][0]
        board[move[0]][move[1]] = game.EMPTY

    move_best = np.argmax(possible_moves)

    for move in legal_moves:
        if move[1] == move_best:
            return move, possible_moves[move_best]
    
    return legal_moves[0], possible_moves[0]


def get_board_by_player (board, player_color):

    observation = [[[0 for i in range(8)] for j in range(8)] for k in range(2)]

    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x] == player_color:
                observation[0][y][x] = 1
            elif board[y][x] == -player_color:
                observation[1][y][x] = 1

    return observation

def self_play (model, test=False):

    current_player = FIRST_PLAYER
    result = 0
    board = game.reset_board()
    first_game_records = []
    second_game_records = []
    last_moves = None

    for turn in range(42):
        
        if test:
            if current_player == FIRST_PLAYER:
                move, q = search(model, board, current_player)
            else:
                print(np.array(board))
                y = input("input row:")
                x = input("input column:")
                move = [int(y), int(x)]
        else:
            legal_moves = game.get_legal_moves(board)
            move = legal_moves[random.randrange(0, len(legal_moves))]

        board[move[0]][move[1]] = current_player
        result = game.check_win(board, 4)

        if current_player == FIRST_PLAYER:
            record = [get_board_by_player(board, FIRST_PLAYER), 0]
            first_game_records.append(record) 
        else:
            record = [get_board_by_player(board, SECOND_PLAYER), 0]
            second_game_records.append(record)

        current_player *= -1

        if result != 0:
            break

    r = random.choice(first_game_records)
    r[1] = 1 if result == FIRST_PLAYER else -1 if result == SECOND_PLAYER else 0
    loss = model.train(np.array(r[0]).reshape(1, 8, 8, 2), np.array(r[1]).reshape(1,1), 1)
    
    r = random.choice(second_game_records)
    r[1] = -1 if result == FIRST_PLAYER else 1 if result == SECOND_PLAYER else 0
    loss = model.train(np.array(r[0]).reshape(1, 8, 8, 2), np.array(r[1]).reshape(1,1), 1)

    return loss, result

def play ():
    model_path = "model_trained_on_random_data.h5"
    model = NN(model_path)

    count = 0
    total_loss = 0

    while True:
        count += 1
        loss, result = self_play(model, test=False)
        total_loss += loss
        print(total_loss / count, count)
        if (count + 1) % 1000 == 0:
            model.model.save("model_temp.h5")

if __name__ == "__main__":
    play()
    