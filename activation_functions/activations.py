"""
Custom and standard activation functions
"""
import tensorflow as tf
from tensorflow import keras


def get_activation(name):
    """
    Get activation function by name
    
    Args:
        name: String name of activation function
        
    Returns:
        Activation function or layer
    """
    name = name.lower()
    
    if name == 'relu':
        return keras.activations.relu
    elif name == 'sigmoid':
        return keras.activations.sigmoid
    elif name == 'tanh':
        return keras.activations.tanh
    elif name == 'softmax':
        return keras.activations.softmax
    elif name == 'swish':
        return keras.activations.swish
    elif name == 'gelu':
        return keras.activations.gelu
    elif name == 'elu':
        return keras.activations.elu
    elif name == 'selu':
        return keras.activations.selu
    elif name == 'leaky_relu':
        return keras.layers.LeakyReLU(alpha=0.2)
    elif name == 'prelu':
        return keras.layers.PReLU()
    elif name == 'mish':
        return mish
    elif name == 'custom_new':
        # Your new activation function here
        return custom_new_activation
    else:
        raise ValueError(f"Activation function {name} not supported")


@tf.function
def mish(x):
    """
    Mish activation function
    Mish: A Self Regularized Non-Monotonic Activation Function
    https://arxiv.org/abs/1908.08681
    
    f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    """
    return x * tf.math.tanh(tf.math.softplus(x))


@tf.function
def custom_new_activation(x):
    """
    Custom new activation function - Example template
    
    This is a placeholder for your new activation function.
    Replace this with your proposed activation function.
    
    Example ideas:
    - Combination of existing activations
    - Parametric variations
    - Novel mathematical formulations
    
    Current implementation: Modified Swish-like function
    f(x) = x * sigmoid(beta * x) where beta is learnable or fixed
    """
    beta = 1.5  # You can make this learnable
    return x * tf.nn.sigmoid(beta * x)


class LearnableActivation(keras.layers.Layer):
    """
    Learnable activation function with trainable parameters
    
    Example: Parametric activation where parameters are learned during training
    f(x) = alpha * x + beta * activation(gamma * x)
    """
    
    def __init__(self, base_activation='relu', **kwargs):
        super(LearnableActivation, self).__init__(**kwargs)
        self.base_activation = keras.activations.get(base_activation)
        
    def build(self, input_shape):
        # Learnable parameters
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        self.gamma = self.add_weight(
            name='gamma',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        super(LearnableActivation, self).build(input_shape)
    
    def call(self, inputs):
        return self.alpha * inputs + self.beta * self.base_activation(self.gamma * inputs)
    
    def get_config(self):
        config = super(LearnableActivation, self).get_config()
        config.update({
            'base_activation': self.base_activation
        })
        return config


class AdaptiveActivation(keras.layers.Layer):
    """
    Adaptive activation that combines multiple activation functions
    with learnable weights
    
    f(x) = w1*act1(x) + w2*act2(x) + ... + wn*actn(x)
    where sum(wi) = 1 (softmax normalized)
    """
    
    def __init__(self, activations=['relu', 'tanh', 'sigmoid'], **kwargs):
        super(AdaptiveActivation, self).__init__(**kwargs)
        self.activation_names = activations
        self.activations = [keras.activations.get(act) for act in activations]
        
    def build(self, input_shape):
        # Learnable weights for each activation
        self.weights_raw = self.add_weight(
            name='activation_weights',
            shape=(len(self.activations),),
            initializer='ones',
            trainable=True
        )
        super(AdaptiveActivation, self).build(input_shape)
    
    def call(self, inputs):
        # Normalize weights using softmax
        weights = tf.nn.softmax(self.weights_raw)
        
        # Compute weighted sum of activations
        output = tf.zeros_like(inputs)
        for i, act in enumerate(self.activations):
            output += weights[i] * act(inputs)
        
        return output
    
    def get_config(self):
        config = super(AdaptiveActivation, self).get_config()
        config.update({
            'activations': self.activation_names
        })
        return config


# Dictionary mapping for easy access
ACTIVATION_FUNCTIONS = {
    'relu': 'relu',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
    'softmax': 'softmax',
    'swish': 'swish',
    'gelu': 'gelu',
    'elu': 'elu',
    'selu': 'selu',
    'leaky_relu': 'leaky_relu',
    'mish': 'mish',
    'custom_new': 'custom_new'
}