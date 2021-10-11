from keras.layers.core import Activation
from keras.models import Model
from tensorflow.python.ops.variables import model_variables
from keras.saving.hdf5_format import save_weights_to_hdf5_group
from tensorflow.python.training.tracking.util import Checkpoint
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import os
import io
import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import tempfile
import tensorflow as tf
import zipfile
import IPython
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from gym.envs.registration import register
import gym
from tensorflow import keras
from keras.models import Sequential, save_model, load_model
from tensorflow.python.summary.summary_iterator import summary_iterator

tempdir = os.getenv("TEST_TMPDIR", tempfile.gettempdir())

        

# Run dqn with Tetris
def dqn():
    env = Tetris()
    env_name = "tetris.py"
    collect_steps_per_iteration = 100
    replay_buffer_capacity = 100000
    learning_rate = 1e-3
    save_every_episode = 100
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 10
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    global_step = tf.compat.v1.train.get_global_step()


    saved_model_path = "saved_models/second_model.pb"
    saved_model_dir = os.path.dirname(saved_model_path)
    checkpoint_path = "training_3/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    def create_model():
        model = tf.keras.models.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(32,32)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10)
        ]) 

        model.compile(optimizer='adam',
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.SparseCategoricalAccuracy()])
        return model

    
    def save_weights(self, name):
        self.model.save_weights(name)

    def load_weights(self, name):
        self.model.load_weights(name)

    latest = tf.train.latest_checkpoint(checkpoint_dir)

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0

        #LOAD LAST MODEL IF EPISODE == 1
        if episode == 0:
            model = load_model(saved_model_path.format(epoch=1))
            agent.model = model
            print("\n\n !!!!!!!!!!!!!!!!!!!!LOAD MODEL, loaded weightss:  ")
            print(model.layers[0].weights) 

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render,
                                    render_delay=render_delay)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)
            model = agent._build_model
            agent.model = model
            best_score = max(scores)
           # model = create_model()
            #model.fit()
            print("\n\n ******AGENT TRAINDED, BEST SCORE: ", best_score, log)

        ##############ANOTHER FUNCTIONS TO COLLECT DATA######################
        #register(id='Tetris', entry_point)
        #train_py_env = suite_gym.load(env_name)
        #eval_py_env = suite_gym.load(env_name)

        #train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        #eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        #replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
         #   data_spec=agent.collect_data_spec,
          #  batch_size=train_env.batch_size,
           # max_length=replay_buffer_capacity)

        #collect_driver = dynamic_step_driver.DynamicStepDriver(
         #   train_env,
          #  agent.collect_policy,
           # observers=[replay_buffer.add_batch],
            #num_steps=collect_steps_per_iteration)
        #####################################################################

        #Save 
        if episode % save_every_episode == 0:
            model = agent._build_model()
            save_model(model, saved_model_path.format(epoch=1))
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
            model.save_weights(checkpoint_path.format(epoch=1))
            print("\n\n ******SAVING AGENT, sample results:  ")
            print(model.layers[0].weights)
            #print(model.layers[0].bias.numpy())
            
            


        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                    max_score=max_score)


if __name__ == "__main__":
    dqn()
