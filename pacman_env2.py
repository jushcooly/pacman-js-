import pygame
import numpy as np
import random
from collections import deque 

#Constant
BLOCK_SIZE = 20
ROWS = 23
COLS = 21
SCREEN_WIDTH = COLS * BLOCK_SIZE
SCREEN_HEIGHT = ROWS * BLOCK_SIZE
FPS = 60  

# Color
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
FOOD_COLOR = (254, 184, 151)

#Direction
DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT = 0, 1, 2, 3
DX = [0, 0, -1, 1]
DY = [-1, 1, 0, 0]

# MAP
RAW_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1],
    [1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 0, 0, 0], 
    [1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1],
    [1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1],
    [1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

class PacmanEnv:
    def __init__(self):
        self.raw_map = np.array(RAW_MAP)
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Pacman RL Environment")
        self.clock = pygame.time.Clock()
        self.HUNGER_THRESHOLD = 150
        
        self.enable_ghost = True 
        self.tutorial_mode = False
        
        self.reset()

    def reset(self):
        self.grid = self.raw_map.copy() 
        self.pacman_pos = [1, 1] 
        self.score = 0
        self.done = False
        self.steps = 0
        
        self.prev_pos = None  
        self.steps_since_last_food = 0
        
        self.visit_map = np.zeros((ROWS, COLS))
        
        if self.enable_ghost:
            self.ghost_pos = [9, 10] 
        else:
            self.ghost_pos = [-10, -10]

        return self.get_state()
    #move ghost(using bfs)
    def _move_ghost_smart(self):
        if not self.enable_ghost: return
        if not hasattr(self, 'ghost_prob'): self.ghost_prob = 0.8 
        if random.random() < self.ghost_prob: return

        start = (self.ghost_pos[0], self.ghost_pos[1])
        target = (self.pacman_pos[0], self.pacman_pos[1])
        if start == target: return

        queue = deque([(start, [])]) 
        visited = set([start])

        while queue:
            (curr_x, curr_y), path = queue.popleft()
            if (curr_x, curr_y) == target:
                if path: self.ghost_pos = list(path[0])
                return
            for i in range(4):
                nx, ny = curr_x + DX[i], curr_y + DY[i]
                if 0 <= nx < COLS and 0 <= ny < ROWS:
                    if self.grid[ny][nx] != 1 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [(nx, ny)]))

    def step(self, action):
        self.steps += 1
        self.steps_since_last_food += 1
        
        nx = self.pacman_pos[0] + DX[action]
        ny = self.pacman_pos[1] + DY[action]
        
        
        reward = -0.01

        #check the wall and boundery
        if nx < 0 or nx >= COLS or ny < 0 or ny >= ROWS:
             reward -=1 
        elif self.grid[ny][nx] == 1:
            reward -= 1 
        else:
            # move penalty when pecman doesn't move
            if self.prev_pos is not None and [nx, ny] == self.prev_pos:
                reward -= 0.2
            self.prev_pos = self.pacman_pos
            self.pacman_pos = [nx, ny]
            
            # Recommend Eulerian path
            visit_count = self.visit_map[ny][nx]
            if visit_count > 0:
                revisit_penalty = min(2, visit_count * 0.5)
                reward -= revisit_penalty
            self.visit_map[ny][nx] += 1
            
            # prey
            if self.grid[ny][nx] == 2:
                reward += 10 
                self.score += 10
                self.grid[ny][nx] = 3 
                self.steps_since_last_food = 0
                self.visit_map[ny][nx] = 0 
        
        # logic of ghost
        if self.enable_ghost:
            self._move_ghost_smart() 
            if self.pacman_pos == self.ghost_pos:
                self.done = True
                if self.score >= 1000: reward -= 10 #make the reward according to score
                elif self.score >= 500: reward -= 30
                else: reward -= 50 

            dist_to_ghost = abs(self.pacman_pos[0] - self.ghost_pos[0]) + abs(self.pacman_pos[1] - self.ghost_pos[1])
            if self.ghost_prob < 1.0 and 2 <= dist_to_ghost <= 4:
                reward += 1.0 

        # when clear
        if not np.any(self.grid == 2):
            reward = 1000 
            self.done = True
            #get the reward bonus when the tutorial(None ghost mode) is finished faaster
            if self.tutorial_mode:
                time_bonus = max(0, (6000 - self.steps) * 0.1)
                reward += time_bonus

        # hunger penalty
        if self.steps_since_last_food > self.HUNGER_THRESHOLD:
            hunger_penalty = 0.1 * ((self.steps_since_last_food - self.HUNGER_THRESHOLD) / 50)
            reward -= hunger_penalty
        #time limited
        if self.tutorial_mode:
            if self.steps >= 6000:
                self.done = True
                #penalty according to score and remained food
                remaining_food = np.sum(self.grid == 2)
                penalty = remaining_food * 10
                reward -= penalty
        
        #When the mode is not tutorial limiting step is 4000
        else:
            if self.steps > 4000: 
                self.done = True
                reward -= 50 
            
        return self.get_state(), reward, self.done, {}

    def get_state(self):# the function describing the state of pecman 
        state = [
            self.pacman_pos[0] / COLS, self.pacman_pos[1] / ROWS,
            self.ghost_pos[0] / COLS, self.ghost_pos[1] / ROWS
        ]
        food_indices = np.argwhere(self.grid == 2)
        if len(food_indices) > 0:
            mean_y, mean_x = food_indices.mean(axis=0)
            state.extend([(mean_x - self.pacman_pos[0]) / COLS, (mean_y - self.pacman_pos[1]) / ROWS])
        else:
            state.extend([0, 0])
            
        nearest_food_dir = [0, 0]
        min_dist = float('inf')
        for y in range(ROWS):
            for x in range(COLS):
                if self.grid[y][x] == 2:
                    dist = abs(self.pacman_pos[0] - x) + abs(self.pacman_pos[1] - y)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_food_dir = [(x - self.pacman_pos[0]) / COLS, (y - self.pacman_pos[1]) / ROWS]
        state.extend(nearest_food_dir)
        
        px, py = self.pacman_pos
        wall_sensor = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            sx, sy = px + dx, py + dy
            if sx < 0 or sx >= COLS or sy < 0 or sy >= ROWS or self.grid[sy][sx] == 1:
                wall_sensor.append(1.0)
            else:
                wall_sensor.append(0.0)
        state.extend(wall_sensor) 
        return np.array(state, dtype=np.float32)

    def render(self):
        pygame.event.pump()
        if pygame.display.get_active():
            self.screen.fill(BLACK)
            for y in range(ROWS):
                for x in range(COLS):
                    rect = (x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
                    if self.grid[y][x] == 1:
                        pygame.draw.rect(self.screen, BLUE, rect)
                    elif self.grid[y][x] == 2:
                        pygame.draw.circle(self.screen, FOOD_COLOR, 
                                         (x * BLOCK_SIZE + BLOCK_SIZE//2, y * BLOCK_SIZE + BLOCK_SIZE//2), 3)
            px, py = self.pacman_pos
            pygame.draw.circle(self.screen, YELLOW, 
                             (px * BLOCK_SIZE + BLOCK_SIZE//2, py * BLOCK_SIZE + BLOCK_SIZE//2), BLOCK_SIZE//2 - 2)
            if self.enable_ghost:
                gx, gy = self.ghost_pos
                pygame.draw.rect(self.screen, RED, 
                            (gx * BLOCK_SIZE + 2, gy * BLOCK_SIZE + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))
            pygame.display.flip()
        self.clock.tick(FPS)