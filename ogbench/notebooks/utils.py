from typing import final
import numpy as np
from tqdm import tqdm
import jax
from utils.evaluation import supply_rng
import matplotlib.pyplot as plt
##========== END IMPORTS ==========##

def all_cells(env):
    all_cells = []
    maze_map = env.unwrapped.maze_map
    for i in range(maze_map.shape[0]):
        for j in range(maze_map.shape[1]):
            if maze_map[i, j] == 0:
                x, y = env.unwrapped.ij_to_xy(np.array([i, j]))
                all_cells.append((x, y))

    return all_cells


def rollout(env, goals, agent, num_steps=500, temperature=0.2, seed=0):
    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(seed))
    ob, _ = env.reset()
    traj = [ob[:2].copy()]
    for i in tqdm(range(num_steps)):
        action = actor_fn(observations=ob, temperature=temperature, goals=goals)
        action = np.clip(action, -1, 1)
        next_observation, _, _, _, _ = env.step(action)
        traj.append(next_observation[:2])
        ob = next_observation  
    return traj  

def rollout_fn(env, goals, actor_fn, num_steps=500, temperature=0.2, seed=0):
    # actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(seed))
    def check_radius(ob, goals, radius=1):
        if goals is None:
            return False
        # if goals.ndim > 1:
        return np.any(np.linalg.norm(goals[:, :2] - ob[:2], axis=1) < radius)
        # else:
            
        #     return np.linalg.norm(goals[:2] - ob[:2]) < radius
    ob, _ = env.reset()
    traj = [ob[:2].copy()]
    goal_reached = False
    for i in tqdm(range(num_steps)):
        action = actor_fn(observations=ob, temperature=temperature, goals=goals)
        action = np.clip(action, -1, 1)
        next_observation, _, _, _, _ = env.step(action)
        traj.append(next_observation[:2])
        ob = next_observation
        goal_reached = goal_reached or check_radius(ob, goals)  
    final_ob_near_goal = check_radius(ob, goals)
    # final_ob_near_goal = True
    return {'traj' : traj, 'goal_reached': goal_reached, 'final_ob_near_goal': final_ob_near_goal}

def plot_axes(ax, traj, goals, all_cells):
    if goals is not None and goals.ndim > 1:
        ax.scatter(*zip(*goals[:, :2]), c='red', s=50, label='Goal')
    else:
        ax.scatter(*goals[:2], c='red', s=50, label='Goal')

    ax.scatter(*zip(*traj), c=range(len(traj)), cmap='viridis', s=10)
    ax.scatter(*traj[0], c='blue', s=50, label='Start')
    ax.scatter(*zip(*all_cells), c='gray', s=1, label='Cells')

def get_goals_list(env, goal_coords):
    ob, _ = env.reset()

    goals = []
    for goal_coord in goal_coords:
        goal = ob.copy()
        goal[:2] = goal_coord
        goals.append(goal)
    goals = np.array(goals)

    return goals