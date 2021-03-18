import random
from copy import copy

import gym
import numpy as np
import scipy.misc as misc
from gym.wrappers.time_limit import TimeLimit
import PIL
from PIL import Image

def make_env(env_id):
    original_name = env_id
    short = False
    if '-v' in env_id:
        env_id = env_id[:env_id.index('-v')].lower()
    if env_id.startswith('short'):
        env_id = env_id[len('short'):]
        short = True
    # Use our task_by_name function to get the env
    return task_by_name(env_id, original_name, short)

def task_by_name(name, original_name=None, short=False):
    if name == "reacher":
        return reacher(short=short)
    elif name == "humanoid":
        return humanoid()
    elif name == "hopper":
        return hopper(short=short)
    elif name in ["walker"]:
        return walker(short=short)
    elif name == "swimmer":
        return swimmer()
    elif name == "ant":
        return ant()
    elif name in ["cheetah", "halfcheetah"]:
        return cheetah(short=short)
    elif name in ["pendulum"]:
        return pendulum()
    elif name in ["doublependulum"]:
        return double_pendulum()
    else:
        try:
            # If an original_name is provided, try to make an environment from that.
            # See "make_env"
            env = gym.make(original_name)

            if name == "montezumarevenge":
                nav_to_start_commands = [
                    # Basic start:
                    [],
                    # Start at the base of the ladder in the bottom right of starting room:
                    # [11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11,
                    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                ]
                env = RandomStartPoint(env, nav_to_start_commands)
                env = DoneOnLivesBelowThreshold(env, 5)

            env = CompressedPixelViewer(env)
            env = PixelEnvViewer(env, flip=True, fps=30)
            env = limit(t=500, env=env)
            return env
        except Exception:
            raise ValueError(name)

class TransparentWrapper(gym.Wrapper):
    """Passes missing attributes through the wrapper stack"""

    def __getattr__(self, attr):
        parent = super()
        if hasattr(parent, attr):
            return getattr(parent, attr)
        if hasattr(self.env, attr):
            return getattr(self.env, attr)
        # HACK: Gym wraps environments with a TimeLimit wrapper sometimes, hiding the the ArcadeLearningEnvironment (ALE).
        # We want to reach past the wrapper to get it. (clone_state is a helper function for the ALE. See atari_env.py in gym.)
        if (attr in ['ale', 'clone_state', 'restore_state']) \
                and hasattr(self.env, 'env') and self.env is not None and hasattr(self.env.env, attr):
            return getattr(self.env.env, attr)
        raise AttributeError(type(self), attr)

class MjViewer(TransparentWrapper):
    """Adds a space-efficient human_obs to info that allows rendering videos subsequently"""

    def __init__(self, env, fps=40):
        self.fps = fps
        super().__init__(env)

    def _get_full_obs(self):
        return (copy(self.env.model.data.qpos[:, 0]), copy(self.env.model.data.qvel[:, 0]))

    def _set_full_obs(self, obs):
        qpos, qvel = obs[0], obs[1]
        self.env.set_state(qpos, qvel)

    def render_full_obs(self, full_obs):
        old_obs = self._get_full_obs()
        self._set_full_obs(full_obs)
        self._get_viewer().render()
        data, width, height = self._get_viewer().get_image()
        result = ((width, height, 3), data)
        self._set_full_obs(old_obs)
        return result

    def _step(self, a):
        human_obs = self._get_full_obs()
        ob, reward, done, info = self.env._step(a)
        info["human_obs"] = human_obs
        return ob, reward, done, info

class PixelEnvViewer(TransparentWrapper):
    """Saves the raw human_obs to info that allows rendering videos subsequently"""

    def __init__(self, env, flip=False, fps=40):
        self.fps = fps
        self.flip = flip
        super().__init__(env)

    def render_full_obs(self, full_obs):
        return full_obs  # Don't need to do anything because we already have the source pixels!

    def _step(self, a):
        ob, reward, done, info = self.env._step(a)
        if self.flip:
            info["human_obs"] = np.flip(ob, axis=0)
        else:
            info["human_obs"] = ob
        return ob, reward, done, info

class CompressedPixelViewer(TransparentWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = np.eye(84)  # HACK: This is just so the .shape == (84,84)

    def _rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def _preprocess(self, image):
        image = self._rgb2gray(image)
        #image = misc.imresize(image, [84, 84], 'bilinear')
        image = np.array(Image.fromarray(image).resize((84,84),resample=PIL.Image.BILINEAR))
        image = image.astype(np.float32) / 128.0 - 1.0
        return image

    def _reset(self):
        ob = self.env.reset()
        ob = self._preprocess(ob)
        return ob

    def _step(self, a):
        ob, reward, done, info = self.env._step(a)
        ob = self._preprocess(ob)
        return ob, reward, done, info

class RandomStartPoint(TransparentWrapper):
    def __init__(self, env, nav_to_start_commands):
        super().__init__(env)
        self._reset_options = [self._build_reset_option(nav_to_start)
                               for nav_to_start in nav_to_start_commands]

    def _build_reset_option(self, nav_to_start):
        """ Takes a list of instructions and returns a saved environment state. """
        self.env._reset()
        for command in nav_to_start:
            self.env.step(command)
        return self.clone_state()

    def _reset(self):
        self.env.reset()  # <-- This is VITAL for resetting the inner TimeLimit. Otherwise all episodes will finish instantly after 10000 global steps.
        start_point = random.choice(self._reset_options)
        # There's a BUG in Gym: https://github.com/openai/gym/commit/339415aa03a9b039a51f67798a44f8cd21464091#diff-56ee07e2f2b6161a732de3552602ba98
        # Use this to route around it for the moment:
        state_ref = self.ale.decodeState(start_point)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)
        # TODO: Once it's fixed in the package manager, use this:
        # self.restore_state(start_point)
        ob, _, _, _ = self.env.step(0)  # Kick it off with a NULLOP
        return ob

class DoneOnLivesBelowThreshold(TransparentWrapper):
    def __init__(self, env, threshold):
        super().__init__(env)
        self.threshold = threshold

    def _step(self, a):
        ob, reward, done, info = super()._step(a)
        return ob, reward, done or self.ale.lives() < self.threshold, info

class UseReward(TransparentWrapper):
    """Use a reward other than the normal one for an environment.
     We do this because humans cannot see torque penalties
     """

    def __init__(self, env, reward_info_key):
        self.reward_info_key = reward_info_key
        super().__init__(env)

    def _step(self, a):
        ob, reward, done, info = super()._step(a)
        return ob, info[self.reward_info_key], done, info

class NeverDone(TransparentWrapper):
    """Environment that never returns a done signal"""

    def __init__(self, env, bonus=lambda a, data: 0.):
        self.bonus = bonus
        super().__init__(env)

    def _step(self, a):
        ob, reward, done, info = super()._step(a)
        bonus = self.bonus(a, self.env.model.data)
        reward = reward + bonus
        done = False
        return ob, reward, done, info

class TimeLimitTransparent(TimeLimit, TransparentWrapper):
    pass

def limit(env, t):
    return TimeLimitTransparent(env, max_episode_steps=t)

def reacher(short=False):
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    env = mujoco.ReacherEnv()
    env = UseReward(env, reward_info_key="reward_dist")
    env = MjViewer(fps=10, env=env)
    return limit(t=20 if short else 50, env=env)

def hopper(short=False):
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    bonus = lambda a, data: (data.qpos[1, 0] - 1) + 1e-3 * np.square(a).sum()
    env = mujoco.HopperEnv()
    env = MjViewer(fps=40, env=env)
    env = NeverDone(bonus=bonus, env=env)
    env = limit(t=300 if short else 1000, env=env)
    return env

def humanoid(standup=True, short=False):
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    env = mujoco.HumanoidEnv()
    env = MjViewer(env, fps=40)
    env = UseReward(env, reward_info_key="reward_linvel")
    if standup:
        bonus = lambda a, data: 5 * (data.qpos[2, 0] - 1)
        env = NeverDone(env, bonus=bonus)
    return limit(env, 300 if short else 1000)

def double_pendulum():
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    bonus = lambda a, data: 10 * (data.site_xpos[0][2] - 1)
    env = mujoco.InvertedDoublePendulumEnv()
    env = MjViewer(env, fps=10)
    env = NeverDone(env, bonus)
    env = limit(env, 50)
    return env

def pendulum():
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher

    # bonus = lambda a, data: np.concatenate([data.qpos, data.qvel]).ravel()[1] - 1.2
    def bonus(a, data):
        angle = data.qpos[1, 0]
        return -np.square(angle)  # Remove the square of the angle

    env = mujoco.InvertedPendulumEnv()
    env = MjViewer(env, fps=10)
    env = NeverDone(env, bonus)
    env = limit(env, 25)  # Balance for 2.5 seconds
    return env

def cheetah(short=False):
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    env = mujoco.HalfCheetahEnv()
    env = UseReward(env, reward_info_key="reward_run")
    env = MjViewer(env, fps=20)
    env = limit(env, 300 if short else 1000)
    return env

def swimmer(short=False):
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    env = mujoco.SwimmerEnv()
    env = UseReward(env, reward_info_key="reward_fwd")
    env = MjViewer(env, fps=40)
    env = limit(env, 300 if short else 1000)
    return env

def ant(standup=True, short=False):
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    env = mujoco.AntEnv()
    env = UseReward(env, reward_info_key="reward_forward")
    env = MjViewer(env, fps=20)
    if standup:
        bonus = lambda a, data: data.qpos.flat[2] - 1.2
        env = NeverDone(env, bonus)
    env = limit(env, 300 if short else 1000)
    return env

def walker(short=False):
    from gym.envs import mujoco  # Import here to avoid forcing the user to have MuJoCo to run Teacher
    bonus = lambda a, data: data.qpos[1, 0] - 2.0 + 1e-3 * np.square(a).sum()
    env = mujoco.Walker2dEnv()
    env = MjViewer(env, fps=30)
    env = NeverDone(env, bonus)
    env = limit(env, 300 if short else 1000)
    return env
