import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import *
from rl.memory import SequentialMemory

from heatmap_environment import HeatmapEnv as selected_environment


ENV_NAME = 'heatmap'

env = selected_environment()
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))


# 100q on 16k, 14k, 15k, 13k, 19k, 13k, 19k, 19k, 14k
# Valid on 24k, 23k
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(96))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

nb_max_episode_steps = env.num_steps_per_episode
env.steps_before_rendering = 55000
nb_steps = 60000

weights_name = 'dqn_heatmap_weights'
if_learn = True

memory = SequentialMemory(limit=nb_steps, window_length=1)

# 0.1 : 4k, 0.25 : 4k, 0.5 : 7k-inf
policy_06_14_16_20 = BoltzmannGumbelQPolicy(C = 20.0) # 20, 50
# more stable
# 0.1 : 4k, 0.25 : 4-5k, 0.5 : 10k-inf
policy_06_13_19_00 = BoltzmannQPolicy(tau = 1.0) # 0.5
policy_06_14_16_15 = MaxBoltzmannQPolicy(eps = 0.5)
policy = policy_06_14_16_20

target_model_update_06_05_20_49 = 1e-2
target_model_update_06_05_22_18 = 1e-1
target_model_update_06_13_19_07 = 1e-3
target_model_update = target_model_update_06_05_20_49
bactch_size_06_05_22_18 = 32
bactch_size_07_05_16_07 = 64
batch_size = bactch_size_06_05_22_18
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, 
	enable_dueling_network=True, dueling_type='avg',
	target_model_update=target_model_update, policy=policy,
	batch_size = 32)

lr_06_05_20_49 = 1e-3
lr_06_05_22_18 = 1e-2
lr_06_13_19_07 = 5e-4
lr = lr_06_05_20_49
dqn.compile(Adam(lr=lr), metrics=['mae'])

if if_learn:
	dqn.fit(env, nb_steps=nb_steps, visualize=True, verbose=2,
	 nb_max_episode_steps=nb_max_episode_steps)

	dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
else:
	dqn.load_weights(weights_name + '.h5f')	
	env.steps_before_rendering = 0

dqn.test(env, nb_episodes = 30, visualize=True, nb_max_episode_steps=nb_max_episode_steps)