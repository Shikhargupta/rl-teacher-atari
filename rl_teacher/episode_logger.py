from copy import deepcopy
import os.path as osp
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

class EpisodeLogger(tf.summary.FileWriter):
    """Tracks and logs agent performance"""

    def __init__(self, name, timesteps_per_summary=int(1e3)):
        logs_path = osp.expanduser('~/tb/rl-teacher/%s' % (name))
        super().__init__(logs_path)

        self.summary_count = 0
        self.timesteps_per_summary = timesteps_per_summary

        self._timesteps_elapsed = 0
        self._timesteps_since_last_summary = 0

        self.last_n_scores = deque(maxlen=100)

    @property
    def timesteps_elapsed(self):
        return self._timesteps_elapsed

    def log_episode(self, path):
        path_length = len(path["obs"])
        self._timesteps_elapsed += path_length
        self._timesteps_since_last_summary += path_length

        if 'new' in path:  # PPO puts multiple episodes into one path
            path_count = np.sum(path["new"])
            for _ in range(path_count):
                self.last_n_scores.append(np.sum(path["original_rewards"]).astype(float) / path_count)
        else:
            self.last_n_scores.append(np.sum(path["original_rewards"]).astype(float))

        if self._timesteps_since_last_summary >= self.timesteps_per_summary:
            self.summary_count += 1
            self.log_simple("agent/true_reward_per_episode", np.mean(self.last_n_scores))
            self.log_simple("agent/total_steps", self._timesteps_elapsed)
            self._timesteps_since_last_summary -= self.timesteps_per_summary
            self.flush()

    def log_simple(self, tag, simple_value, debug=False):
        self.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=simple_value)]), self.summary_count)
        if debug:
            print("%s    =>    %s" % (tag, simple_value))
