import numpy as np

from gym import spaces
from gym import Env

import time
import math

from gym.envs.classic_control import rendering


WARNING = "\033[93m"
ENDC = '\033[0m'


class HeatmapEnv(Env):
	def __init__(self):
		self.steps_before_rendering = 0
		self.good_enough_z = -3.95
		self.step_width = 0.1
		self.num_steps_per_episode = 100
		
		self.num_steps_to_keep = 5
		self.num_steps_to_check = 5
		self.num_z_to_keep = 3

		self.stay_penalty = -2.0
		self.step_penalty = -6.0
		self.edge_penalty = -10.0
		self.path_non_consistency_penalty = -7.0 / self.num_steps_to_check
		self.opt_reward = 10.0

		self.shift_amount_x = 1
		self.shift_amount_y = 1
		self.shift_period = 3000

		self.low = np.array([6., 3.])
		self.high = np.array([8., 5.])
		self.z_low = 0.0
		self.z_high = -4.0

		# previous actions
		self.low = np.concatenate(( self.low, np.zeros(self.num_steps_to_keep * 5) ))
		self.high = np.concatenate(( self.high, np.ones(self.num_steps_to_keep * 5) ))

		# previous z
		self.low = np.concatenate(( self.low, np.zeros(self.num_z_to_keep) ))
		self.high = np.concatenate(( self.high, np.ones(self.num_z_to_keep) ))

		# shift (temperature)
		self.low = np.concatenate(( self.low, np.zeros(1) ))
		self.high = np.concatenate(( self.high, np.ones(1) ))

		self.action_space = spaces.Discrete(5)
		self.observation_space = spaces.Box(low=self.low, high=self.high)

		self.allowed_x = np.arange(self.low[0], self.high[0] + self.step_width, self.step_width)
		self.allowed_y = np.arange(self.low[1], self.high[1] + self.step_width, self.step_width)
		self.diagonal_length = np.linalg.norm(np.array((self.allowed_x[0], self.allowed_y[0])) 
			- np.array((self.allowed_x[-1], self.allowed_y[-1])))
		self.space_length = np.array([self.high[0] - self.low[0], self.high[1] - self.low[1]])

		self.viewer = None

		self.x = 0.0
		self.y = 0.0
		self.z = 0.0
		self.total_steps = 0
		self.last_steps = []
		self.episode_actions = []
		self.last_z = []
		self.shift_value = 0.0
		self.shift_x = 0.0
		self.shift_y = 0.0

		print(self.__class__.__name__, 'inited!')

	def step(self, action):
		self.total_steps += 1
		self.episode_actions.append(action)

		self.last_steps.append(action)
		if len(self.last_steps) > self.num_steps_to_keep:
			self.last_steps.pop(0)

		if_done = False
		reward = 0.0
		penalty = 0.0

		if action == 0:
			if not self.x - self.step_width < self.allowed_x[0]:
				self.x -= self.step_width
			else:
				penalty += self.edge_penalty
		if action == 1:
			if not self.x + self.step_width > self.allowed_x[-1]:
				self.x += self.step_width
			else:
				penalty += self.edge_penalty
		if action == 2:
			if not self.y - self.step_width < self.allowed_y[0]:
				self.y -= self.step_width
			else:
				penalty += self.edge_penalty
		if action == 3:
			if not self.y + self.step_width > self.allowed_y[-1]:
				self.y += self.step_width
			else:
				penalty += self.edge_penalty
		
		if not action == 4:
			reward += self.step_penalty
		else:
			reward += self.stay_penalty

		self.z = self.equal_pits_and_one_opt(self.x, self.y)

		self.last_z.append((self.z - self.z_low) / (self.z_high - self.z_low))
		if len(self.last_z) > self.num_z_to_keep:
			self.last_z.pop(0)

		if 0 in self.last_steps and 1 in self.last_steps[-self.num_steps_to_check:]:
			reward += self.path_non_consistency_penalty
		if 2 in self.last_steps and 3 in self.last_steps[-self.num_steps_to_check:]:
			reward += self.path_non_consistency_penalty

		if self.z <= self.good_enough_z:
			reward += self.opt_reward

		return self._get_obs(), reward, if_done, {}

	def equal_pits_and_one_opt(self, x, y):
		minor_pits = [[4, 4], [4, 7], [7, 4], [7, 7]]
		opt_pit_idx = 2
		opt_pit_inner_r = 0.25
		outer_r = 1.2
		inner_r = 0.5

		self.shift_value = math.sin((self.total_steps) * math.pi * 2 / self.shift_period) / 2
		self.shift_x = self.shift_value * self.shift_amount_x
		self.shift_y = self.shift_value * self.shift_amount_y

		minor_pits[opt_pit_idx][0] += self.shift_x
		minor_pits[opt_pit_idx][1] += self.shift_y

		resulting_field_value = 0.0
		for idx, pit in enumerate(minor_pits):
			r = np.linalg.norm(np.array([x, y]) - np.array(pit) + np.array([0.2, -0.1]))
			if r > outer_r:
				r = float('inf')
			if idx is opt_pit_idx:
				if r < opt_pit_inner_r:
					r = opt_pit_inner_r
			else:
				if r < inner_r:
					r = inner_r
			resulting_field_value -= 1 / r

		return resulting_field_value

	def reset(self):
		print("%.2f" % self.x, "%.2f" % self.y)
		self.x = self.allowed_x[np.random.randint(0, self.allowed_x.size)]
		self.y = self.allowed_y[np.random.randint(0, self.allowed_y.size)]
		self.num_steps = 0

		self.last_steps = []
		for i in range(self.num_steps_to_keep):
			self.last_steps.append(-1)

		self.last_z = []
		for i in range(self.num_z_to_keep):
			self.last_z.append(0)

		print(self.episode_actions[:-self.num_steps_to_check], self.episode_actions[-self.num_steps_to_check:])
		self.episode_actions = []
		return self._get_obs()

	def _get_obs(self):
		last_steps = np.array(self.last_steps)
		one_hot = np.zeros((last_steps.size, 5))
		one_hot[np.arange(last_steps.size), last_steps] = 1
		for i in np.arange(last_steps.size):
			if last_steps[i] < 0:
				one_hot[i] = np.zeros(5)
		one_hot = one_hot.flatten()
		shift = [self.shift_value + 0.5]
		return np.concatenate(( ((np.array([self.x, self.y]) - np.array([self.low[0], self.low[1]]))
			/ self.space_length), one_hot, self.last_z, shift))

	def close(self):
		pass

	def render(self, mode='human'):
		if self.total_steps < self.steps_before_rendering:
			return None
		screen_width = 400
		screen_height = 400

		x_threshold = 40

		world_width = x_threshold * 2
		scale = screen_width / world_width
		cartwidth = 30.0
		cartheight = 30.0

		if self.viewer is None:
			self.viewer = rendering.Viewer(screen_width, screen_height)

			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			self.cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			self.cart.add_attr(self.carttrans)
			self.viewer.add_geom(self.cart)

		if self.last_steps is None: return None

		norm_x_y = ((np.array([self.x, self.y]) - np.array([self.low[0], self.low[1]]))
			/ self.space_length)
		self.carttrans.set_translation(norm_x_y[0] * screen_width, norm_x_y[1] * screen_height)
		if self.z < self.good_enough_z:
			self.cart.set_color(0, 1., 0.0)	
		else:
			self.cart.set_color(1., 0.0, 0.0)
			
		return self.viewer.render(return_rgb_array = mode=='rgb_array')