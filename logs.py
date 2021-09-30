from keras.callbacks import TensorBoard
#from tensorflow.summary import FileWriter
import tensorflow as tf
from keras.models import filewriter

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    #Problematic function in new version of libs
    def log(self, step, **stats):
       # self._write_logs(stats, step)
        self.writer = filewriter(self.log_dir)  
