import pygame
import torch
import os
import time
import numpy as np
import cv2  
from pacman_env2 import PacmanEnv
from ddqn_agent import DDQNAgent


MODEL_PATH = "pacman_best.pth"  
TEST_EPISODES = 5            
GHOST_DIFFICULTY = 0.2          
RECORD_VIDEO = True             
VIDEO_DIR = "videos"          

def test():
    env = PacmanEnv()
    state_dim = 12
    action_dim = 4
    
    agent = DDQNAgent(state_dim, action_dim)

    
    if os.path.exists(MODEL_PATH):
        agent.policy_net.load_state_dict(
            torch.load(MODEL_PATH, map_location=agent.device)
        )
        print(f"Model created ({MODEL_PATH})")
    else:
        print(f"File is not exsited: {MODEL_PATH}")
        return


    if RECORD_VIDEO and not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print(f" '{VIDEO_DIR}' Folder created")

    agent.epsilon = 0.0
    agent.policy_net.eval()
    
    print(f" TEST START!! (trial: {TEST_EPISODES}  , Recording: {RECORD_VIDEO})")

    for episode in range(TEST_EPISODES):
        state = env.reset()
        env.ghost_prob = GHOST_DIFFICULTY
        
        done = False
        step = 0
        
        
        video_writer = None
        if RECORD_VIDEO:
            
            screen_width = env.screen.get_width()
            screen_height = env.screen.get_height()
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            filename = os.path.join(VIDEO_DIR, f"episode_{episode+1}.mp4")
            
            video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (screen_width, screen_height))

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            
            env.render()
            
            
            if RECORD_VIDEO and video_writer is not None:
                
                view = pygame.surfarray.array3d(env.screen)
                
                
                view = view.transpose([1, 0, 2])
            
                frame = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
            
            state = next_state
            step += 1
            time.sleep(0.01) 
        if video_writer is not None:
            video_writer.release()
            print(f"   Saving is completed!: {filename}")

        print(f"Ep {episode+1} Result | Score: {env.score} | steos: {step}")

    print("🏁 Test and Recording is finished!")
    pygame.quit()

if __name__ == "__main__":
    test()