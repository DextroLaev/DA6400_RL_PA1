# Reinforcement Learning PA1: Q-Learning and SARSA for CartPole and MountainCar

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4.1-orange)](https://github.com/google/jax)
[![NumPy](https://img.shields.io/badge/NumPy-1.22.0-blue)](https://numpy.org/)

This repository contains the implementation and analysis of Q-Learning and SARSA algorithms applied to two classic reinforcement learning environments: CartPole-v1 and MountainCar-v0. The project was completed as part of the DA6400 Reinforcement Learning course.

## Project Overview
This project empirically evaluates two fundamental Reinforcement Learning (RL) algorithms—SARSA and Q-Learning—across three Gymnasium environments with varying complexity:
- **CartPole-v1**: Balance a pole on a moving cart
- **MountainCar-v0**: Drive a car up a steep hill using momentum
- **MiniGrid-Dynamic-Obstacles-5x5-v0**: Navigate through moving obstacles to reach a goal

Key contributions include:
- Implementation of both on-policy (SARSA) and off-policy (Q-Learning) approaches
- Comparative analysis of exploration strategies: ε-greedy vs Softmax
- Hyperparameter tuning for optimal performance
- Robust evaluation across 5 random seeds

## Algorithms Implemented
### Q-Learning (Off-Policy)
```python
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```
- Uses Softmax exploration policy
- Learns optimal policy independently of behavior policy

### SARSA (On-Policy)
```python
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```
- Uses ε-greedy exploration policy
- Learns action values for current policy

### CartPole-V1 Optimal Parameters
| Algorithm          | Learning Rate (α) | τ_start/ε_start | τ_end/ε_end |
|--------------------|------------------|-----------------|------------|
| Q-Learning (Softmax) | 0.05            | 3.0             | 0.2        |
|                    | 0.5             | 3.0             | 0.1        |
|                    | 0.1             | 3.0             | 0.2        |
| SARSA (ε-greedy)   | 0.2             | 0.9             | 0.05       |
|                    | 0.2             | 1.0             | 0.05       |
|                    | 0.2             | 0.8             | 0.05       |

### MountainCar-V0 Optimal Parameters
| Algorithm          | Learning Rate (α) | τ_start/ε_start | τ_end/ε_end |
|--------------------|------------------|-----------------|------------|
| Q-Learning (Softmax) | 0.1             | 1.0             | 0.02       |
|                    | 0.2             | 1.0             | 0.02       |
|                    | 0.1             | 1.0             | 0.01       |
| SARSA (ε-greedy)   | 0.1             | 0.9             | 0.01       |
|                    | 0.05            | 0.7             | 0.01       |
|                    | 0.05            | 0.7             | 0.05       |

## Key Findings

1. **CartPole Environment**:
   - Q-Learning with moderate learning rates (α=0.1) performed best
   - Higher initial exploration (ε_start=0.9-1.0) was crucial for SARSA
   - Temperature decay (τ) needed to be gradual for stable learning

2. **MountainCar Environment**:
   - Required more aggressive exploration initially
   - Lower final exploration rates (ε_end=0.01) led to better convergence
   - Learning rates needed to be smaller (α=0.05-0.1) for stable learning

## Code Structure
## Getting Started
### Prerequisites
- Python 3.8+
- JAX
- Gymnasium
- NumPy

### Installation
```bash
pip install -r requirements.txt
```

### Running Experiments (for example: cartpole)


```bash
  cd Jax
  cd cartpole
```
```bash
  python main.py
```

## Team Members
- Shuvrajee Das [DA24D402] (IIT Madras, DSAI Dept)  
  [da24d402@smail.iitm.ac.in] | [shuvrajeet17@gmail.com]

- Rajshekhar Rakshit [CS24S031]  (IIT Madras, CSE Dept)  
  [cs24s031@smail.iitm.ac.in] | [rajshekharrakshit123@gmail.com]

## References
1. Barto, Sutton, Anderson (1983) - Neuronlike adaptive elements
2. Moore (1990) - Efficient memory-based learning for robot control
