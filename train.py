import pygame
import torch
import os
import numpy as np
from pacman_env2 import PacmanEnv
from ddqn_agent import DDQNAgent

EPISODES = 3000
MAX_STEPS = 4000
SYNC_INTERVAL = 10

MAX_TUTORIAL_STEPS = 6000 

def main():
    env = PacmanEnv()
    
    ghost_start_episode = 200
    adjusted_score = 0    
    
    start_prob = 1.0             
    final_ghost_prob = 0.2       
    #ghost difficulty propotional with episode
    ghost_decay_rate = (start_prob - final_ghost_prob) / (EPISODES - ghost_start_episode)
    current_ghost_prob = start_prob 

    state_dim = 12
    action_dim = 4
    
    best_score = -float('inf') 
    agent = DDQNAgent(state_dim, action_dim)
    
    
    agent.epsilon = 0.99
    
    #linear decrease when tutorial ends
    epsilon_start_real = 0.8
    epsilon_end = 0.01
    epsilon_decay_steps = 2800 
    epsilon_decrement = (epsilon_start_real - epsilon_end) / epsilon_decay_steps
    
    render_mode = False 
    scores_history = [] 
    
    print(" (Double DQN)")
    print(f" First tutorial : {ghost_start_episode} round  is tutorial.")
    print(f" (Max tutorial step: {MAX_TUTORIAL_STEPS} )")
    print(" 'V' key: video mode on and off")

    for episode in range(EPISODES):
        if episode == ghost_start_episode:
            print("\n" + "="*40)
            print(f"End of tutorial")
            print("="*40 + "\n")
            scores_history = []  
            best_score = -float('inf') 
            agent.epsilon = epsilon_start_real

        if episode < ghost_start_episode:
            env.enable_ghost = False
            env.tutorial_mode = True   
        else:
            env.enable_ghost = True
            env.tutorial_mode = False  
        
        state = env.reset()
        env.ghost_prob = current_ghost_prob
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            #when the tutorial, the step is over 6000, end this game
            if env.tutorial_mode:
                if step >= MAX_TUTORIAL_STEPS:
                    break
            else:
                # the real mode(limitation: 4000)
                if step >= MAX_STEPS:
                    break

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_v:
                        render_mode = not render_mode
                        print(f"📺  Video mode: {'ON' if render_mode else 'OFF'}")

            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step() 
            
            if render_mode: env.render()
            else: pygame.event.pump()
            
            state = next_state
            total_reward += reward
            step += 1
        
        if episode >= ghost_start_episode:
            if current_ghost_prob > final_ghost_prob:
                current_ghost_prob -= ghost_decay_rate 

        # fast decrease when tutorial mode
        if episode < ghost_start_episode:
           
            if agent.epsilon > 0.05:
                agent.epsilon *= 0.995
        else:
            #decrease slowly when the mode is real
            if agent.epsilon > epsilon_end:
                agent.epsilon -= epsilon_decrement
            else:
                agent.epsilon = epsilon_end

        if episode % SYNC_INTERVAL == 0:
            agent.update_target_network()

        scores_history.append(env.score)
        avg_score = np.mean(scores_history[-100:]) if scores_history else 0

        mode_str = "TUTORIAL" if episode < ghost_start_episode else "REAL"
        
        print(f"Ep: {episode+1} [{mode_str}] | Score: {env.score:.0f} | Step: {step} | Avg(100): {avg_score:.1f} | GhostProb: {current_ghost_prob:.2f} | Eps: {agent.epsilon:.2f} | {'Y' if render_mode else 'N'}")
        
        if episode >= ghost_start_episode and current_ghost_prob <= 0.4:
            if env.score > best_score:
                best_score = env.score
                adjusted_score = best_score
                torch.save(agent.policy_net.state_dict(), "pacman_best.pth")
                print(f" ((Score: {best_score})")

        if (episode + 1) % 100 == 0:
            torch.save(agent.policy_net.state_dict(), f"pacman_ep{episode+1}.pth")

    print("\n" + "="*30)
    print("Final training report")
    print(f"Total Episode : {EPISODES}")
    if not scores_history: scores_history = [0]
    print(f" Max Score : {(adjusted_score)}")
    print(f" Total avg score : {np.mean(scores_history):.1f}")
    print("="*30)
    torch.save(agent.policy_net.state_dict(), "pacman_final.pth")
    pygame.quit()

if __name__ == "__main__":
    main()