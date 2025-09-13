"""
RL GAME (Unified scenarios)
Run examples:
  python main.py --scenario eat --train --num-enemies 3 --enemy-speed 3
  python main.py --scenario avoid --train --num-enemies 6 --enemy-speed 15
"""
import argparse
import pickle
from Env import Env_enemy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["eat", "avoid"], default="eat")
    parser.add_argument("--train", action="store_true", help="Train mode; omit for eval")
    parser.add_argument("--num-enemies", type=int, default=3)
    parser.add_argument("--enemy-speed", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    env = Env_enemy(
        TRAIN=args.train,
        scenario=args.scenario,
        num_enemies=args.num_enemies,
        enemy_speed=args.enemy_speed,
    )
    reward_history = []
    t = 0
    while True:
        t += 1
        print(f"Episode: {t} | scenario={args.scenario} | train={args.train}")
        reward_history.append(env.total_reward)
        if env.TRAIN:
            # save the reward history and model every 10 episodes
            if t % 10 == 0:
                # save the model
                if args.scenario == "eat":
                    env.agent.model.save("model/model_eat.h5")
                elif args.scenario == "avoid":
                    env.agent.model.save("model/model_avoid.h5")
                print(f"Model saved for scenario: {args.scenario}")

                if args.scenario == "eat":
                    # save the reward history
                    with open("plot/reward_history_eat.pickle", "wb") as f:
                        pickle.dump(reward_history, f)
                    # save the q_values
                    with open("plot/q_values_eat.pickle", "wb") as f:
                        pickle.dump(env.agent.q_values, f)
                elif args.scenario == "avoid":
                    # save the reward history
                    with open("plot/reward_history_avoid.pickle", "wb") as f:
                        pickle.dump(reward_history, f)
                    # save the q_values
                    with open("plot/q_values_avoid.pickle", "wb") as f:
                        pickle.dump(env.agent.q_values, f)
        env.run()
