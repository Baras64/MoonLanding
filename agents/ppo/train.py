
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(f"{SCRIPT_DIR[:-11]}/sim/")

from environments import get_train_env
from discrete_action_agent import PPOAgent

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create environment')
    parser.add_argument('--envname', metavar='path', required=True,
                        help='Environment name should be [discreteaction-deterministic-fo, discreteaction-stochastic-fo]')

    args = parser.parse_args()
    # with mlflow.start_run():
    #     mlflow.log_param("envname", args.envname)
    env = get_train_env(args.envname)
    
    agent = PPOAgent(env, args.envname)
    agent.run_batch()
    # agent.run()