"""
evaluate.py

This script is used to evaluate the performance of a trained model in different environments.

Usage:
    python evaluate.py --model_path <path_to_model> --env <environment_name>

Example:
    python evaluate.py --agent optimalcontrol --env continuousaction-deterministic-fo
    python evaluate.py --agent optimalcontrol --env continuousaction-stochastic-fo
Arguments:
    --env: The name of the environment in which to evaluate the model.
    --agent: The agent to evaluate the plan.
"""
import argparse
import numpy as np
import json
from tqdm import tqdm
from environments import get_eval_env, env_options
from agents import BaseAgent, OptimalControlAgent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, choices=list(env_options.keys()), required=True)
    parser.add_argument("--agent", type=str, choices=["optimalcontrol", "hybridastar", "dqn", "ppo"], required=True)
    parser.add_argument("--episodes", type=int, required=False, default=10000)

    return parser.parse_args()

def evaluate(agent, env, args):
    scores = [] # list of scores from each episode
    for _ in tqdm(range(args.episodes), "Episode: "):
        state, info = env.reset()
        
        done = False
        total = 0
        while not done:
            a = agent.action(state)
            state, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            total += reward
        scores.append(total)
    return scores

def main():
    args = parse_args()
    print("Evaluating {} agent in {} environment".format(args.agent, args.env))

    try:
        env = get_eval_env(args.env)
        if args.agent == "optimalcontrol":
            agent: BaseAgent = OptimalControlAgent()
            scores = evaluate(agent, env, args)
        elif args.agent == "hybridastar":
            raise NotImplementedError
        elif args.agent == "dqn":
            raise NotImplementedError
        elif args.agent == "ppo":
            raise NotImplementedError
        
        # Save scores in a file as json with file name as env-agent
        with open(f"{args.env}-{args.agent}.json", "w") as f:
            json.dump(scores, f)
        
        print(f"Average Score: {np.mean(scores)}" )
    finally:
        env.close()

if __name__ == "__main__":
    main()