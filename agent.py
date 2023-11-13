import gymnasium as gym
import numpy as np
import heapq

class Node:
    def __init__(self, state):
        self.state = state
        self.parent = None
        self.g = 0
        self.h = 0

    def f(self):
        return self.g + self.h

def a_star_lunar_lander(env):
    start_state = custom_state_representation(env.reset())
    goal_state = np.array([0, 0, 0, 0, 0, 0, 0, 0])

    open_list = []
    closed_list = set()

    start_node = Node(start_state)
    goal_node = Node(goal_state)

    heapq.heappush(open_list, (start_node.f(), start_node))

    while open_list:
        _, current_node = heapq.heappop(open_list)

        if custom_state_compare(current_node.state, goal_node.state):
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            path = path[::-1]
            return path

        closed_list.add(tuple(current_node.state))

        for action in range(env.action_space.n):
            result = env.step(action)

            print("Result tuple:", result)

            next_state, reward, done, _, _ = result

            # Update state representation
            custom_state = custom_state_representation(next_state)

            # Check if the updated state is in the closed list
            if tuple(custom_state) in closed_list:
                continue

            neighbor = Node(custom_state)
            neighbor.parent = current_node
            neighbor.g = current_node.g + 1
            neighbor.h = np.linalg.norm(np.array(neighbor.state) - goal_node.state)

            if tuple(custom_state) not in [item[1].state for item in open_list]:
                heapq.heappush(open_list, (neighbor.f(), neighbor))

            # Check if the episode is done after taking the action
            if done:
                return None  # or handle termination as needed

    return None




def custom_state_representation(state):
    state_array = state[0]  # Extract the array part
    
    # Check if state_array is iterable
    if hasattr(state_array, '__iter__'):
        # Extract individual components from the state_array
        x, y, vx, vy, angle, v_angle, left_leg, right_leg = state_array[:8]
        return np.array([x, y, vx, vy, angle, v_angle, left_leg, right_leg])
    else:
        # Handle the case when state_array is not iterable
        return np.array([state_array])

def custom_state_compare(state1, state2):
    return np.all(np.isclose(state1, state2, atol=0.01))

env = gym.make('LunarLander-v2')
path = a_star_lunar_lander(env)