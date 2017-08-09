import os
import argparse
from time import time

import numpy as np
import tensorflow as tf
from parallel_trpo.train import train_parallel_trpo
from pposgd_mpi.run_mujoco import train_pposgd_mpi
from ga3c.Server import Server as Ga3cServer
from ga3c.Config import Config as Ga3cConfig

from rl_teacher.reward_predictors import TraditionalRLRewardPredictor, ComparisonRewardPredictor
from rl_teacher.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from rl_teacher.envs import get_timesteps_per_episode
from rl_teacher.envs import make_env
from rl_teacher.label_schedules import LabelAnnealer, ConstantLabelSchedule
from rl_teacher.video import SegmentVideoRecorder
from rl_teacher.segment_sampling import segments_from_rand_rollout
from rl_teacher.summaries import AgentLogger, make_summary_writer
from rl_teacher.utils import slugify

def make_comparison_predictor(env, experiment_name, predictor_type, summary_writer,
                              clip_length, n_pretrain_labels, n_labels=None):
    agent_logger = AgentLogger(summary_writer)

    if n_labels:
        label_schedule = LabelAnnealer(
            agent_logger,
            final_timesteps=num_timesteps,
            final_labels=n_labels,
            pretrain_labels=n_pretrain_labels)
    else:
        print("No label limit given. We will request one label every few seconds.")
        label_schedule = ConstantLabelSchedule(pretrain_labels=n_pretrain_labels)

    if predictor_type == "synth":
        comparison_collector = SyntheticComparisonCollector()
    elif predictor_type == "human":
        comparison_collector = HumanComparisonCollector(env, experiment_name=experiment_name)
    else:
        raise ValueError("Bad value for --predictor: %s" % predictor_type)

    return ComparisonRewardPredictor(
        env,
        experiment_name,
        summary_writer,
        comparison_collector=comparison_collector,
        agent_logger=agent_logger,
        label_schedule=label_schedule,
        clip_length=clip_length
    )

def pretrain_predictor(predictor, env_id, n_pretrain_labels, n_pretrain_iters, clip_length, workers):
    predictor.comparison_collector.clear_old_data()

    print("Starting random rollouts to generate pretraining segments. No learning will take place...")
    pretrain_segments = segments_from_rand_rollout(
        env_id, make_env, n_desired_segments=n_pretrain_labels * 2,
        clip_length_in_seconds=clip_length, workers=args.workers)

    # Add segments to comparison collector
    for seg in pretrain_segments:
        predictor.comparison_collector.add_segment(seg)
    # Turn our random segments into comparisons
    for _ in range(n_pretrain_labels):
        predictor.comparison_collector.invent_comparison()
    # Label our comparisons
    predictor.comparison_collector.label_unlabeled_comparisons(goal=n_pretrain_labels, verbose=True)

    # Pretrain predictor
    for i in range(n_pretrain_iters):
        predictor.train_predictor()  # Train on pretraining labels
        if i % 25 == 0:
            print("%s/%s predictor pretraining iters... " % (i, n_pretrain_iters))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--env_id', required=True)
    parser.add_argument('-p', '--predictor', required=True)
    parser.add_argument('-n', '--name', required=True)
    parser.add_argument('-s', '--seed', default=1, type=int)
    parser.add_argument('-w', '--workers', default=4, type=int)
    parser.add_argument('-l', '--n_labels', default=None, type=int)
    parser.add_argument('-L', '--pretrain_labels', default=None, type=int)
    parser.add_argument('-t', '--num_timesteps', default=5e6, type=int)
    parser.add_argument('-a', '--agent', default="ga3c", type=str)
    parser.add_argument('-i', '--pretrain_iters', default=10000, type=int)
    parser.add_argument('-b', '--starting_beta', default=0.1, type=float)
    parser.add_argument('-c', '--clip_length', default=1.5, type=float)
    parser.add_argument('-V', '--no_videos', action="store_true")
    parser.add_argument('-R', '--restore', action="store_true")
    args = parser.parse_args()

    print("Setting things up...")
    env_id = args.env_id
    experiment_name = slugify(args.name)
    run_name = "%s/%s-%s" % (env_id, experiment_name, int(time()))
    summary_writer = make_summary_writer(run_name)
    env = make_env(env_id)
    num_timesteps = int(args.num_timesteps)

    os.makedirs('checkpoints/reward_model', exist_ok=True)
    os.makedirs('segments', exist_ok=True)

    # Make predictor
    if args.predictor == "rl":
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        n_pretrain_labels = args.pretrain_labels if args.pretrain_labels else args.n_labels // 4
        predictor = make_comparison_predictor(
            env, experiment_name, args.predictor, summary_writer, args.clip_length, n_pretrain_labels, args.n_labels)

        if args.restore:
            predictor.load_model_from_checkpoint()
            print("Model loaded from checkpoint!")
        else:
            pretrain_predictor(predictor, env_id, n_pretrain_labels, args.pretrain_iters, args.clip_length, args.workers)

    # Wrap the predictor to capture videos every so often:
    if not args.no_videos:
        video_path = os.path.join('/tmp/rl_teacher_vids', run_name)
        predictor = SegmentVideoRecorder(predictor, env, save_dir=video_path, checkpoint_interval=100)

    print("Starting joint training of predictor and agent")
    if args.agent == "ga3c":
        Ga3cConfig.NETWORK_NAME = experiment_name
        Ga3cConfig.SAVE_FREQUENCY = 200
        Ga3cConfig.LOAD_CHECKPOINT = args.restore
        Ga3cConfig.BETA_START = args.starting_beta
        Ga3cConfig.BETA_END = args.starting_beta * 0.1
        Ga3cConfig.ATARI_GAME = env
        Ga3cConfig.AGENTS = args.workers
        Ga3cServer(predictor).main()
    elif args.agent == "parallel_trpo":
        train_parallel_trpo(
            env_id=env_id,
            make_env=make_env,
            predictor=predictor,
            summary_writer=summary_writer,
            workers=args.workers,
            runtime=(num_timesteps / 1000),
            max_timesteps_per_episode=get_timesteps_per_episode(env),
            timesteps_per_batch=8000,
            max_kl=0.001,
            seed=args.seed,
        )
    elif args.agent == "pposgd_mpi":
        train_pposgd_mpi(lambda: make_env(env_id), num_timesteps=num_timesteps, seed=args.seed, predictor=predictor)
    else:
        raise ValueError("%s is not a valid choice for args.agent" % args.agent)

if __name__ == '__main__':
    main()
