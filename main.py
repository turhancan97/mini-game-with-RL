"""
RL GAME - Two cases: eat and avoid and two observation types: vector and pixels
Run examples:
  python main.py --scenario eat --train --num-enemies 4 --enemy-speed 10 --max-steps 1000 --seed 42 --headless --obs-type pixels
  python main.py --scenario avoid --train --num-enemies 8 --enemy-speed 10 --max-steps 800 --seed 42 --headless --obs-type pixels
"""
import argparse
import os
import pickle
import random
import numpy as np
import tensorflow as tf
from Env import Env_enemy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["eat", "avoid"], default="eat")
    parser.add_argument("--train", action="store_true", help="Train mode; omit for eval")
    parser.add_argument("--num-enemies", type=int, default=3)
    parser.add_argument("--enemy-speed", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step cap per episode")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--headless", action="store_true", help="Run without rendering for faster training")
    parser.add_argument("--obs-type", choices=["vector", "pixels"], default="vector", help="Observation type: low-dim vector or raw pixels")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Ensure output directories exist
    os.makedirs("model", exist_ok=True)
    os.makedirs("plot", exist_ok=True)

    # Set seeds for reproducibility if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
    env = Env_enemy(
        TRAIN=args.train,
        scenario=args.scenario,
        num_enemies=args.num_enemies,
        enemy_speed=args.enemy_speed,
        max_steps=args.max_steps,
        headless=args.headless,
        obs_type=args.obs_type,
    )
    reward_history = []
    t = 0
    while True:
        t += 1
        print(f"Episode: {t} | scenario={args.scenario} | train={args.train}")
        # Run one episode
        env.run()

        # Record episode total reward AFTER episode completes
        reward_history.append(env.total_reward)

        # Decay epsilon once per episode
        env.agent.adaptiveEGreedy()

        if env.TRAIN:
            # save the reward history and model every 10 episodes
            if t % 10 == 0:
                # save the model (separate filenames for CNN vs MLP)
                if args.scenario == "eat":
                    model_path = "model/model_pixels_eat.h5" if args.obs_type == "pixels" else "model/model_vector_eat.h5"
                elif args.scenario == "avoid":
                    model_path = "model/model_pixels_avoid.h5" if args.obs_type == "pixels" else "model/model_vector_avoid.h5"
                env.agent.model.save(model_path)
                print(f"Model saved for scenario: {args.scenario}")

                if args.scenario == "eat":
                    if args.obs_type == "pixels":
                        plot_path = "plot/reward_history_pixels_eat.pickle"
                        qvalues_path = "plot/q_values_pixels_eat.pickle"
                    else:
                        plot_path = "plot/reward_history_vector_eat.pickle"
                        qvalues_path = "plot/q_values_vector_eat.pickle"
                    with open(plot_path, "wb") as f:
                        pickle.dump(reward_history, f)
                    with open(qvalues_path, "wb") as f:
                        pickle.dump(env.agent.q_values, f)
                elif args.scenario == "avoid":
                    if args.obs_type == "pixels":
                        plot_path = "plot/reward_history_pixels_avoid.pickle"
                        qvalues_path = "plot/q_values_pixels_avoid.pickle"
                    else:
                        plot_path = "plot/reward_history_vector_avoid.pickle"
                        qvalues_path = "plot/q_values_vector_avoid.pickle"
                    with open(plot_path, "wb") as f:
                        pickle.dump(reward_history, f)
                    with open(qvalues_path, "wb") as f:
                        pickle.dump(env.agent.q_values, f)
