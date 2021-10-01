from numpy.lib.function_base import _create_arrays
from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
from logs import CustomTensorBoard
from tqdm import tqdm
import os

from tensorflow import keras
from keras.models import Sequential, save_model, load_model
import tensorflow as tf

# Run dqn with Tetris
def dqn():

    weigth_name = './tetris2Dmodel.h5'
    env = Tetris()
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 2000
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 100
    log_every = 5
    replay_start_size = 512
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    def save_weights(self, name):
        self.model.save_weights(name)
    
    def load_weights(self, name):
        self.model.load_weights(name)

    #model = tf.keras.

    load_model
    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    load_weights
    #tf.keras.models.tf.keras.models.load_model(
     #   filePathString,
      #  custom_objects=None, compile=True)
    
   # model.load_weights(latest)
    print("\n\n ******MODEL AND WEIGHTS LOADED*****")

    def create_model():
        model = tf.keras.model



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

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)
            best_score = max(scores)
            print("\n\n ******AGENT TRAINED best score***** :", best_score, log)
            save_weights
            save_model
            #model.save()
            print("\n\n ******MODEL AND WEIGHTS SAVED*****")
        
        # Best score
        #if epsilon_stop_episode == 20:
            


        # Logs
        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            log.log(episode, avg_score=avg_score, min_score=min_score,
                max_score=max_score)

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
