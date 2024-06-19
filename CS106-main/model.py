import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np 
import sys

from tensorflow import keras
print(tf.__version__)
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim): 
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size # Get batch sample to train
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)
    
    def _build_model(self,num_layers, width):
        inputs = keras.Input(shape=(self._input_dim,))

        x = layers.Dense(512, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        
        outputs = layers.Dense(self._output_dim, activation='linear')(x)

        model = keras.Model(inputs = inputs, outputs = outputs, name="my_model")
        model.compile(loss=losses.MeanSquaredError(),  optimizer=Adam(learning_rate=self._learning_rate))
        return model
    
    def predict_one(self, state):
        state = np.reshape(state, [1, self._input_dim])
        # print("state", state)
        return self._model.predict(state)
    
    def predict_batch(self,states):
        # print("stats", states)
        return self._model.predict(states)
    
    def train_batch(self, states, q_sa):
        self._model.fit(states, q_sa, epochs=1, verbose=0)

    def save_model(self, path):
        self._model.save(os.path.join(path, 'trained_model.keras'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size
    
class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.keras')
        # print(model_file_path)
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return self._input_dim