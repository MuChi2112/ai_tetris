import torch
import random
import numpy as np
from collections import deque
import game
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

STATE_NUM = game.col * game.row + len(game.shapes)*2

class Agent:
    def __init__(self, epsilon_decay=0.995, min_epsilon=0.1, load_model=None):
        self.n_games = 0
        self.epsilon = 1.0  # 初始随机性
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(STATE_NUM, 1024, 4)
        if load_model is not None:
            self.model.load_state_dict(torch.load(load_model))
            self.model.eval()  # 将模型设置为评估模式
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self):
        
        temp = [0] * STATE_NUM
        
        for y in range(game.row):
            for x in range(game.col):
                if (x, y) in game.locked_positions:
                    temp[(y * 10) + (x + 1) - 1] = 1

        for j in range(len(game.shapes)):
            if game.shapes[j] == game.current_piece.shape:
                temp[(game.col * game.row) +  j] = 1

        for j in range(len(game.shapes)):
            if game.shapes[j] == game.next_piece.shape:
                temp[(game.col * game.row) + len(game.shapes) +  j] = 1

        piece_pos = game.convert_shape_format(game.current_piece)
        
        for (x, y) in piece_pos:
            if y >= 0:
                temp[(y * 10) + (x + 1) - 1] = 1
        
        return temp

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, train_model=True):
        
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0,0]

        piece_pos = game.convert_shape_format(game.current_piece)
        
        for (x, y) in piece_pos:
            if y < 0:
                final_move[1] = 1
                return final_move

        if random.random() < self.epsilon and train_model==True:
            # print("random")
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            # print("good")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move



def train(max_games=1000, train_model=True):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(load_model='./model/model.pth')

    if not train_model:
        while True:
            state_old = agent.get_state()
            final_move = agent.get_action(state_old, train_model)
            reward, done, score = game.paly_game(final_move)
            if done:
                game.reset()
                agent.n_games += 1
                if score > record:
                    record = score
                    
                print('Game', agent.n_games, 'Score', score, 'Record:', record)
            
    else:
        while agent.n_games < max_games:
            # get old state
            state_old = agent.get_state()
            # print("")
            # print("")
            # for i in range(len(state_old)):
            #     if i>=10 and i%10==0 :
            #         print("")
            #     print(state_old[i], end='')
            

            # get move
            final_move = agent.get_action(state_old,train_model)

            reward, done, score = game.paly_game(final_move)

            # perform move and get new state

            state_new = agent.get_state()

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
                
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                # plot_mean_scores.append(mean_score)
                # plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train(train_model=False)