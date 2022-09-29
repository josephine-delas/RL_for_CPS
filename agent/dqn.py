from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, Sequential, model_from_json, load_model
from tensorflow.keras.losses import Huber

class SimpleNetwork():
  '''
  For the attacker agent, who doesn't need convolution
  '''

  def __init__(self, obs_shape, learning_rate = .2):
    '''
    Ici c'est l'attacker model, il n'a pas vraiment besoin de l'information des états pour savoir quelle action choisir. 
    On ne lui donne donc pas tous les timesteps en input, juste le premier.
    obs_shape = (timesteps, features)
    '''
    self.num_actions=2

    self.model=Sequential()
    self.model.add(Dense(100, input_shape=(obs_shape[1],),
                             activation='relu'))
    self.model.add(Dense(self.num_actions))

    optimizer = optimizers.Adam(0.00025)
    self.model.compile(loss=Huber,optimizer=optimizer)

  def predict(self,state,batch_size):
    """
    state is of shape (batch_size, timesteps, features)
    """
    state = state[:,0,:]
    return self.model.predict(state,batch_size=batch_size)

  def update(self, states, q):
    loss = self.model.train_on_batch(states, q)
    return loss

class QNetwork():
    """
    Q-Network Estimator
    Represents the global model for the table
    """

    def __init__(self,obs_shape,learning_rate=.2):
        """
        obs_shape = (timesteps, features)
        """
        self.num_conv = 10
        self.ker_size = 3
        self.dense_size = 100
        self.num_actions = 2
        
        # Network architecture
        self.model = Sequential()
        self.model.add(Conv1D(self.num_conv, self.ker_size, activation='tanh',input_shape=obs_shape))
        self.model.add(MaxPooling1D(pool_size = 2))
        self.model.add(Conv1D(self.num_conv*2, self.ker_size, activation='tanh', input_shape = self.model.output_shape))
        self.model.add(MaxPooling1D(pool_size = 2))
        self.model.add(Flatten())
        self.model.add(Dense(self.dense_size, activation='relu'))
        self.model.add(Dense(self.num_actions))

        #optimizer = optimizers.SGD(learning_rate)
        optimizer = optimizers.Adam(learning_rate = learning_rate)
        # optimizer = optimizers.RMSpropGraves(learning_rate, 0.95, self.momentum, 1e-2)

        # Compilation of the model with optimizer and loss
        self.model.summary()
        self.model.compile(loss=Huber,optimizer=optimizer)

    def predict(self,state,batch_size):
        """
        Predicts action values.
        state is of shape (None, timesteps, features)
        """
        return self.model.predict(state,batch_size=batch_size)

    def update(self, states, q):
        """
        Updates the estimator with the targets.

        Args:
          states: Target states
          q: Estimated values

        Returns:
          The calculated loss on the batch.
        """
        loss = self.model.train_on_batch(states, q)
        return loss
    
    def copy_model(model):
        """Returns a copy of a keras model."""
        #model.save('tmp_model')
        model.save('temp')
        return load_model('temp')