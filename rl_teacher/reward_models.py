import os
import random
from time import sleep

from multiprocessing import Process

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from keras import backend as K
from scipy import stats

from rl_teacher.clip_manager import SynthClipManager, ClipManager
from rl_teacher.nn import FullyConnectedMLP, SimpleConvolveObservationQNet
from rl_teacher.segment_sampling import segments_from_rand_rollout, sample_segment_from_path, basic_segment_from_null_action
from rl_teacher.utils import corrcoef

def nn_predict_rewards(obs_segments, act_segments, network, obs_shape, act_shape):
    """
    :param obs_segments: tensor with shape = (batch_size, segment_length) + obs_shape
    :param act_segments: tensor with shape = (batch_size, segment_length) + act_shape
    :param network: neural net with .run() that maps obs and act tensors into a (scalar) value tensor
    :param obs_shape: a tuple representing the shape of the observation space
    :param act_shape: a tuple representing the shape of the action space
    :return: tensor with shape = (batch_size, segment_length)
    """
    batchsize = tf.shape(obs_segments)[0]
    segment_length = tf.shape(obs_segments)[1]

    # Temporarily chop up segments into individual observations and actions
    obs = tf.reshape(obs_segments, (-1,) + obs_shape)
    acts = tf.reshape(act_segments, (-1,) + act_shape)

    # Run them through our neural network
    rewards = network.run(obs, acts)

    # Group the rewards back into their segments
    return tf.reshape(rewards, (batchsize, segment_length))

class RewardModel(object):
    def __init__(self, episode_logger):
        self._episode_logger = episode_logger

    def predict_reward(self, path):
        raise NotImplementedError()  # Must be overridden

    def path_callback(self, path):
        self._episode_logger.log_episode(path)

    def train(self, iterations=1, report_frequency=None):
        pass  # Doesn't require training by default

    def save_model_checkpoint(self):
        pass  # Nothing to save

    def try_to_load_model_from_checkpoint(self):
        pass  # Nothing to load

class OriginalEnvironmentReward(RewardModel):
    """Model that always gives the reward provided by the environment."""

    def predict_reward(self, path):
        return path["original_rewards"]

class OrdinalRewardModel(RewardModel):
    """A learned model of an environmental reward using training data that is merely sorted."""

    def __init__(self, model_type, env, env_id, make_env, experiment_name, episode_logger, label_schedule, n_pretrain_clips, clip_length, stacked_frames, workers):
        # TODO It's pretty asinine to pass in env, env_id, and make_env. Cleanup!
        super().__init__(episode_logger)

        if model_type == "synth":
            self.clip_manager = SynthClipManager(env, experiment_name)
        elif model_type == "human":
            self.clip_manager = ClipManager(env, env_id, experiment_name, workers)
        else:
            raise ValueError("Cannot find clip manager that matches keyword \"%s\"" % model_type)

        if self.clip_manager.total_number_of_clips > 0 and not self.clip_manager._sorted_clips:
            # If there are clips but no sort tree, create a sort tree!
            self.clip_manager.create_new_sort_tree_from_existing_clips()
        if self.clip_manager.total_number_of_clips < n_pretrain_clips:
            # If there aren't enough clips, generate more!
            self.generate_pretraining_data(env_id, make_env, n_pretrain_clips, clip_length, stacked_frames, workers)

        self.clip_manager.sort_clips(wait_until_database_fully_sorted=True)

        self.label_schedule = label_schedule
        self.experiment_name = experiment_name
        self._frames_per_segment = clip_length * env.fps
        # The reward distribution has standard dev such that each frame of a clip has expected reward 1
        self._standard_deviation = self._frames_per_segment
        self._elapsed_training_iters = 0
        self._episode_count = 0
        self._episodes_per_training = 50
        self._iterations_per_training = 50
        self._episodes_per_checkpoint = 100

        # Build and initialize our model
        config = tf.compat.v1.ConfigProto(
            # device_count={'GPU': 0},
            # log_device_placement=True,
        )
        config.gpu_options.per_process_gpu_memory_fraction = 0.35  # allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)

        self.obs_shape = env.observation_space.shape
        if stacked_frames > 0:
            self.obs_shape = self.obs_shape + (stacked_frames,)
        self.discrete_action_space = not hasattr(env.action_space, "shape")
        self.act_shape = (env.action_space.n,) if self.discrete_action_space else env.action_space.shape

        self.graph = self._build_model()
        self.sess.run(tf.global_variables_initializer())
        my_vars = tf.global_variables()
        self.saver = tf.train.Saver({var.name: var for var in my_vars}, max_to_keep=0)

    def _build_model(self):
        """Our model takes in path segments with observations and actions, and generates rewards (Q-values)."""
        # Set up observation placeholder
        self.obs_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None) + self.obs_shape, name="obs_placeholder")

        # Set up action placeholder
        if self.discrete_action_space:
            self.act_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None), name="act_placeholder")
            # Discrete actions need to become one-hot vectors for the model
            segment_act = tf.one_hot(tf.cast(self.act_placeholder, tf.int32), self.act_shape[0])
            # HACK Use a convolutional network for Atari
            # TODO Should check the input space dimensions, not the output space!
            net = SimpleConvolveObservationQNet(self.obs_shape, self.act_shape)
        else:
            self.act_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, None) + self.act_shape, name="act_placeholder")
            # Assume the actions are how we want them
            segment_act = self.act_placeholder
            # In simple environments, default to a basic Multi-layer Perceptron (see TODO above)
            net = FullyConnectedMLP(self.obs_shape, self.act_shape)

        # Our neural network maps a (state, action) pair to a reward
        self.rewards = nn_predict_rewards(self.obs_placeholder, segment_act, net, self.obs_shape, self.act_shape)

        # We use trajectory segments rather than individual (state, action) pairs because
        # video clips of segments are easier for humans to evaluate
        self.segment_rewards = tf.reduce_sum(self.rewards, axis=1)

        self.targets = tf.placeholder(dtype=tf.float32, shape=(None,), name="reward_targets")

        self.loss = tf.reduce_mean(tf.square(self.targets - self.segment_rewards))

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        return tf.get_default_graph()

    def predict_reward(self, path):
        """Predict the reward for each step in a given path"""
        with self.graph.as_default():
            predicted_rewards = self.sess.run(self.rewards, feed_dict={
                self.obs_placeholder: np.asarray([path["obs"]]),
                self.act_placeholder: np.asarray([path["actions"]]),
                K.learning_phase(): False
            })
        return predicted_rewards[0]  # The zero here is to get the single returned path.

    def path_callback(self, path):
        super().path_callback(path)
        self._episode_count += 1

        # We may be in a new part of the environment, so we take a clip to learn from if requested
        if self.clip_manager.total_number_of_clips < self.label_schedule.n_desired_labels:
            new_clip = sample_segment_from_path(path, int(self._frames_per_segment))
            if new_clip:
                self.clip_manager.add(new_clip, source="on-policy callback")

        # Train our model every X episodes
        if self._episode_count % self._episodes_per_training == 0:
            self.train(iterations=self._iterations_per_training, report_frequency=25)

        # Save our model every X steps
        if self._episode_count % self._episodes_per_checkpoint == 0:
            self.save_model_checkpoint()

    def generate_pretraining_data(self, env_id, make_env, n_pretrain_clips, clip_length, stacked_frames, workers):
        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        if self.clip_manager.total_number_of_clips == 0:
            # We need a valid clip for the root node of our search tree.
            # Null actions are more likely to generate a valid clip than a random clip from random actions.
            first_clip = basic_segment_from_null_action(env_id, make_env, clip_length, stacked_frames)
            # Add the null-action clip first, so the root is valid.
            self.clip_manager.add(first_clip, source="null-action", sync=True)  # Make synchronous to ensure this is the first clip.
            # Now add the rest

        desired_clips = n_pretrain_clips - self.clip_manager.total_number_of_clips

        random_clips = segments_from_rand_rollout(
            env_id, make_env, n_desired_segments=desired_clips,
            clip_length_in_seconds=clip_length, stacked_frames=stacked_frames, workers=workers)

        for clip in random_clips:
            self.clip_manager.add(clip, source="random rollout")

    def calculate_targets(self, ordinals):
        """ Project ordinal information into a cardinal value to use as a reward target """
        max_ordinal = self.clip_manager.maximum_ordinal  # Equivalent to the size of the sorting tree
        step_size = 1.0 / (max_ordinal + 1)
        offset = step_size / 2
        targets = [self._standard_deviation * stats.norm.ppf(offset + (step_size * o)) for o in ordinals]
        return targets

    def train(self, iterations=1, report_frequency=None):
        self.clip_manager.sort_clips()
        # batch_size = min(128, self.clip_manager.number_of_sorted_clips)
        _, clips, ordinals = self.clip_manager.get_sorted_clips()  # batch_size=batch_size

        obs = [clip['obs'] for clip in clips]
        acts = [clip['actions'] for clip in clips]
        targets = self.calculate_targets(ordinals)

        with self.graph.as_default():
            for i in range(1, iterations + 1):
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.obs_placeholder: np.asarray(obs),
                    self.act_placeholder: np.asarray(acts),
                    self.targets: np.asarray(targets),
                    K.learning_phase(): True
                })
                self._elapsed_training_iters += 1
                if report_frequency and i % report_frequency == 0:
                    print("%s/%s reward model training iters. (Err: %s)" % (i, iterations, loss))
                elif iterations == 1:
                    print("Reward model training iter %s (Err: %s)" % (self._elapsed_training_iters, loss))

    def _checkpoint_filename(self):
        return 'checkpoints/reward_model/%s/treesave' % (self.experiment_name)

    def save_model_checkpoint(self):
        print("Saving reward model checkpoint!")
        self.saver.save(self.sess, self._checkpoint_filename())

    def try_to_load_model_from_checkpoint(self):
        filename = tf.train.latest_checkpoint(os.path.dirname(self._checkpoint_filename()))
        if filename is None:
            print('No reward model checkpoint found on disk for experiment "{}"'.format(self.experiment_name))
        else:
            self.saver.restore(self.sess, filename)
            print("Reward model loaded from checkpoint!")
            # Dump model outputs with errors
            if True:  # <-- Toggle testing with this
                with self.graph.as_default():
                    clip_ids, clips, ordinals = self.clip_manager.get_sorted_clips()
                    targets = self.calculate_targets(ordinals)
                    for i in range(len(clips)):
                        predicted_rewards = self.sess.run(self.rewards, feed_dict={
                            self.obs_placeholder: np.asarray([clips[i]["obs"]]),
                            self.act_placeholder: np.asarray([clips[i]["actions"]]),
                            K.learning_phase(): False
                        })[0]
                        reward_sum = sum(predicted_rewards)
                        starting_reward = predicted_rewards[0]
                        ending_reward = predicted_rewards[-1]
                        print(
                            "Clip {: 3d}: predicted = {: 5.2f} | target = {: 5.2f} | error = {: 5.2f}"  # | start = {: 5.2f} | end = {: 5.2f}"
                            .format(clip_ids[i], reward_sum, targets[i], reward_sum - targets[i]))  # , starting_reward, ending_reward))
