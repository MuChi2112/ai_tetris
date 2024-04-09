import torch
import random
import numpy as np
from collections import deque
import game
from model import Linear_QNet, QTrainer
from helper import plot
import copy


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

STATE_NUM = game.col * game.row + 5 + 7

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


    def to2DBoard(self, locked_positions):
        board = [ [0]*game.col for _ in range(game.row) ]
        for y in range(game.row):
            for x in range(game.col):
                if (x, y) in game.locked_positions:
                    board[y][x] = 1
                    
        # print(board)
        return board

    def calc_heuristics(self, board, x):
        """Calculate heuristics

        The heuristics are composed by: number of holes, number of blocks above
        hole and maximum height.

        """
        total_holes        = 0
        locals_holes       = 0
        blocks_above_holes = 0
        sum_heights        = 0

        for y in range(game.row-1, -1,-1):
            if board[y][x] == 0:
                locals_holes += 1
            else:
                sum_heights += game.row-y

                if locals_holes > 0:
                    total_holes += locals_holes
                    locals_holes = 0

                if total_holes > 0:
                    blocks_above_holes += 1

        return total_holes, blocks_above_holes, sum_heights

    def cal_init(self, board2D):
        total_holes          = 0
        total_blocking_bocks = 0

        for x2 in range(0, game.col):
            b = self.calc_heuristics(board2D, x2)

            total_holes          += b[0]
            total_blocking_bocks += b[1]


        return total_holes, total_blocking_bocks


    def calculate_edges(self, new_board, piece_pos, rows, cols):
        floor_edges = 0
        wall_edges = 0
        block_edges = 0
        
        for x, y in piece_pos:
            # 檢查與地板的連接
            
            if y + 1 == rows:
                floor_edges += 1
            elif new_board[y + 1][x] == 1:
                block_edges += 1
            
            # 檢查與左側牆壁的連接
            if x == 0:
                wall_edges += 1
            elif new_board[y][x - 1] == 1:
                block_edges += 1
                
            # 檢查與右側牆壁的連接
            if x + 1 == cols:
                wall_edges += 1
            elif new_board[y][x + 1] == 1:
                block_edges += 1
                
            # 檢查與上方方塊的連接（僅當y>0時檢查，避免索引出界）
            if y > 0 and new_board[y - 1][x] == 1:
                block_edges += 1

        return floor_edges, wall_edges, block_edges


    def calcu_remove_lines(self, board):
        # 初始化移除行數計數器
        remove_lines = 0

        # 遊戲板的行數
        rows = len(board)
        # 遊戲板的列數
        cols = len(board[0])

        # 從底部向上遍歷每一行
        for i in range(rows-1, -1, -1):
            # 檢查這一行是否已經完全被方塊填滿
            full = True
            for j in range(cols):
                if board[i][j] == 0:  # 假設0代表空位
                    full = False
                    break

            # 如果這一行完全被填滿，則應該被消除
            if full:
                # 從遊戲板中移除這一行
                del board[i]
                # 在遊戲板的頂部加入一新的空白行
                board.insert(0, [0 for _ in range(cols)])
                # 更新移除行數
                remove_lines += 1

        # 返回總共移除的行數
        return remove_lines


        

    def calcu_nxt_move(self, total_holes_bef, total_blocking_bocks_bef):
        grid = game.create_grid(game.locked_positions)
        next_move_piece = copy.deepcopy(game.current_piece)
        
        while game.valid_space(next_move_piece, grid):
            next_move_piece.y+=1
            
        next_move_piece.y-=1
            
        new_board = self.to2DBoard(game.locked_positions)
        piece_pos = game.convert_shape_format(next_move_piece)
        
        floor_edges, wall_edges, block_edges = self.calculate_edges(new_board, piece_pos, game.row, game.col)
        
        for x,y in piece_pos:
            new_board[y][x] = 1
                
        remove_lines = self.calcu_remove_lines(new_board)
        
        
        total_holes          = 0
        total_blocking_bocks = 0
        max_height = 0

        for x2 in range(0, game.col):
            b = self.calc_heuristics(new_board, x2)

            total_holes          += b[0]
            total_blocking_bocks += b[1]
            max_height += b[2]
            
        new_holes = total_holes_bef - total_holes
        new_blocking_bocks = total_blocking_bocks_bef - total_blocking_bocks

        return max_height, remove_lines, new_holes, new_blocking_bocks, block_edges, floor_edges, wall_edges

    def get_state(self):
        
        temp = [0] * STATE_NUM
        
        for y in range(game.row):
            for x in range(game.col):
                if (x, y) in game.locked_positions:
                    temp[(y * 10) + (x + 1) - 1] = 1

        # shape index
        temp[(game.col * game.row) +1 -1 ] =  game.shapes.index(game.current_piece.shape)
        # rotation
        temp[(game.col * game.row) +2 -1 ] = game.current_piece.rotation
        # x
        temp[(game.col * game.row) +3 -1 ] = game.current_piece.x        
        # y
        temp[(game.col * game.row) +4 -1 ] = game.current_piece.y

        # next shape index
        temp[(game.col * game.row) +5 -1 ] = game.shapes.index(game.next_piece.shape)
                
        board2D = self.to2DBoard(game.locked_positions)
        
        total_holes, total_blocking_bocks = self.cal_init(board2D)        
        
        max_height, remove_lines, new_holes, new_blocking_bocks, block_edges, floor_edges, wall_edges = self.calcu_nxt_move(total_holes, total_blocking_bocks)
        
        temp[(game.col * game.row) +6 -1 ] = max_height
        temp[(game.col * game.row) +7 -1 ] = remove_lines
        temp[(game.col * game.row) +8 -1 ] = new_holes
        temp[(game.col * game.row) +9 -1 ] = new_blocking_bocks
        temp[(game.col * game.row) +10 -1 ] = block_edges
        temp[(game.col * game.row) +11 -1 ] = floor_edges
        temp[(game.col * game.row) +12 -1 ] = wall_edges
        
        print(remove_lines)
        
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


    if not train_model:
        agent = Agent(load_model='./model/model.pth')
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
        agent = Agent()
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
    train(train_model=True)