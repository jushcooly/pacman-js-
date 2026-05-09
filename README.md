# AI Pac-Man: RL-driven Autonomous Agent
> **A self-moving Pac-Man agent leveraging Reinforcement Learning and Graph Theory for optimal navigation and survival.**

---

##  Project Overview
This project transforms the classic Pac-Man game into a testbed for advanced **Reinforcement Learning (RL)**. Unlike standard rule-based agents, this Pac-Man learns to navigate complex environments, avoid ghosts, and optimize pellet collection through a combination of deep learning architectures and mathematical path-finding algorithms.

## Core Technologies & Algorithms

### 1. Reinforcement Learning (Deep Q-Learning)
* **Double DQN (DDQN)**: Reduced overestimation of Q-values by decoupling the selection and evaluation of actions, leading to more stable learning.
* **Prioritized Experience Replay (PER)**: Improved sample efficiency by prioritizing transitions with higher TD-error, allowing the agent to learn more effectively from rare but important experiences.

### 2. Behavioral Innovation: The "Hunger" Mechanic
To prevent the agent from playing too conservatively (e.g., hiding in a corner), I implemented a **Hunger System**:
* The reward for eating pellets dynamically increases over time.
* As the "hunger" state grows, the agent is incentivized to take calculated risks to find food, effectively balancing the exploration-exploitation trade-off.

### 3. Path Optimization: Eulerian Trail
* Applied **Graph Theory** to the maze structure.
* By calculating an **Eulerian Trail**, the agent identifies the most efficient path to traverse every corridor, ensuring 100% map coverage with minimal backtracking.

### 4. Adaptive Ghost Intelligence 
To provide a challenging environment for the RL agent, the ghosts are equipped with:
* **BFS Pathfinding**: Ghosts use **Breadth-First Search (BFS)** to calculate the shortest path to Pac-Man in real-time, making their pursuit highly efficient.
* **Dynamic Difficulty Scaling**: The game monitors the learning progress. As the number of episodes increases, the **movement delay of the ghosts is incrementally reduced**, forcing the RL agent to adapt to a faster and more aggressive environment.

## Key Features
* **Ghost Evasion**: Real-time trajectory prediction and avoidance.
* **Dynamic Path Planning**:  switches between real mode and tutorial mode.


