from numpy.lib.function_base import _create_arrays
from tensorflow.python.ops.variables import model_variables
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import os

#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import tf_agents
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from tensorflow.python.summary.summary_iterator import summary_iterator

from tensorflow import keras
from keras.models import Sequential, save_model, load_model
import tensorflow as tf



# Run dqn with Tetris
def dqn():

    weigth_name = './tetris2Dmodel.h5'
    #env = Tetris()
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 2000
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 100
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']



    env_name = "Tetris"
    env = env_name
    tempdir = "trained_agent/agent.txt"

    collect_steps_per_iteration = 100
    replay_buffer_capacity = 100000

    fc_layer_params = (100,)
    learning_rate = 1e-3
    log_interval = 5

    num_eval_episodes = 10
    eval_interval = 1000
    #train_py_env = suite_gym.load(env_name)
    #eval_py_env = suite_gym.load(env_name)

    train_env = tf_agents.environments.suite_gym.load(env)
    eval_env = tf_py_environment.TFPyEnvironment(env)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)
    agent.initialize()

    #def save_weights(self, name):
     #   self.model.save_weights(name)
    
    #def load_weights(self, name):
     #   self.model.load_weights(name)
    
   # def create_model():
    #    model = tf.keras.model

   # def create_model():
    #    model = tf.keras.models.Sequential([
     #   keras.layers.Dense(512, activation='relu', input_shape= n_neurons),
      #  keras.layers.Dropout(0.2),
       # keras.layers.Dense(10)
    #])

     #   model.compile(optimizer='adam',
      #          loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
       #         metrics=[tf.metrics.SparseCategoricalAccuracy()])

        #return model


    #model = create_model()
    


    #model = tf.keras.

   # load_model
    #checkpoint_path = "training_2/cp.ckpt"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    #load_weights
    #tf.keras.models.tf.keras.models.load_model(
     #   filePathString,
      #  custom_objects=None, compile=True)
    
    #latest = tf.train.latest_checkpoint(checkpoint_dir)

    #Print example results
    #print(model.layers[0].weights)
    #print(model.layers[0].bias.numpy())
    #Print example results

    #model.summary()
    
    #model = create_model()
    #model.load_weights(latest)
    
   # model.load_weights(latest)
    #print("\n\n ******MODEL AND WEIGHTS LOADED*****")

    
    #def load_weights(self, name):
     #    self.model.load_weights(name)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    log = CustomTensorBoard(log_dir=log_dir)

    #if os.path.exists(weigth_name):

     #   print("\n\n ******weigth name***** :", weigth_name)
        
      #  filepath = "data.txt"
       # f = open(filepath, "a")
        #f.write(weigth_name)
        #f.close()
        
        #DQNAgent().load_weights(name=weigth_name)
        # agent.model.load_weights('./checkpoint_name')
        #checkpoint = tf.train.Checkpoint(epsilon_stop_episode)
        #checkpoint.restore('./checkpoint_name').assert_consumed()
        #print("\n \n *********Weights are loaded********* \n \n")

    scores = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0
        
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

        #######DODATKOWE FUNKCJE DO ZBIERANIA DANYCH##################################
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration)

        collect_driver.run()

        dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

        iterator = iter(dataset)
        ##############################################################################

        # Train
        if episode % train_every == 0:
            agent.train = common.function(agent.train)

            def train_one_iteration():
                collect_driver.run()
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience)
                iteration = agent.train_step_counter.numpy()
                print('iteration: {0}  loss: {1}'.format(iteration, train_loss.loss))
            #agent.train(batch_size=batch_size, epochs=epochs) #<---original line
            #best_score = max(scores)
            #print("\n\n ******AGENT TRAINED best score***** :", best_score, log)
            #save_weights
            #save_model
            #model.save()
            #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)
            #model.save_weights(checkpoint_path.format(epoch=1))
            #print("\n\n ******MODEL AND WEIGHTS SAVED*****")
        
        # Best score
        #if epsilon_stop_episode == 20:
            


        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])
            log.log(episode, avg_score=avg_score, min_score=min_score,
                max_score=max_score)
            checkpoint_dir = os.path.join(tempdir, 'checkpoint')
            train_checkpointer = common.Checkpointer(
                ckpt_dir=checkpoint_dir,
                max_to_keep=1,
                agent=agent,
                policy=agent.policy,
                replay_buffer=replay_buffer,
            )
        policy_dir = os.path.join(tempdir, 'policy')
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        
        print('TRENOWANIE oraz ZAPIS W PUNKCIE KONTROLNYM...')
        train_one_iteration()
        train_checkpointer.save(global_step)
        tf_policy_saver.save(policy_dir)

    #def save_model_and_weights(agent):
     #   agent.model.save('./checkpoint_name')
      #  best_weights = agent.model.get_weights()
       # print("\n\n ****** best weigth***** :", best_weights)
        #filepath = "data.txt"
        #f = open(filepath, "a")
        #f.write(best_weights)
        #f.close()
        #return best_weights
        

    

     #DQNAgent().save_weights(name=weigth_name)
     #keras.models.save_model('./lastmodel')
    #_ = save_model_and_weights(agent)
    #print("\n  ***Weights are saved***\n")
    



if __name__ == "__main__":
    dqn()
