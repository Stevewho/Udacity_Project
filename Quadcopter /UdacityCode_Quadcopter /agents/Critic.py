from keras import layers, models, optimizers
from keras import backend as K

def resdense(features):
    def unit(i):
        hfeatures = max(4,int(features/4))

        ident = i
        i = layers.Dense(features,activation='tanh')(i)

        ident = layers.Dense(hfeatures)(ident)
        ident = layers.Dense(features)(ident)

        return layers.add([ident,i])
    return unit

class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size


        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states=resdense(64)(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.3)(net_states)
        
        net_states=resdense(32)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.3)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = resdense(64)(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.3)(net_actions)
       

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add the action tensor in the 2nd hidden layer; Combine state and action pathways
        net = layers.concatenate([net_states, net_actions])
        net = resdense(32)(net)

        # Add more layers to the combined network if needed
        
        

        # Add final output layer to prduce action values (Q values)
        Q_values = resdense(1)(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr = 0.01)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)