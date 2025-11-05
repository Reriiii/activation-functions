"""
Base model class for all architectures
"""
import tensorflow as tf
from tensorflow import keras
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all models
    """
    
    def __init__(self, input_shape, num_classes, activation='relu'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.activation = activation
        self.model = None
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    def compile_model(self, optimizer='adam', learning_rate=0.001):
        """
        Compile the model with optimizer and loss function
        
        Args:
            optimizer: Optimizer name or instance
            learning_rate: Learning rate for optimizer
        """
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer
            
        self.model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
    def get_model(self):
        """Return the compiled model"""
        if self.model is None:
            self.build_model()
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        self.model.summary()
        
    def count_parameters(self):
        """Count total and trainable parameters"""
        if self.model is None:
            self.build_model()
            
        trainable_count = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        non_trainable_count = sum([tf.size(w).numpy() for w in self.model.non_trainable_weights])
        
        return {
            'total': trainable_count + non_trainable_count,
            'trainable': trainable_count,
            'non_trainable': non_trainable_count
        }


class ConvBlock(keras.layers.Layer):
    """
    Reusable convolutional block with Conv2D -> BatchNorm -> Activation
    """
    
    def __init__(self, filters, kernel_size, strides=1, padding='same', 
                 activation='relu', use_bias=False, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        
        self.conv = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias
        )
        self.bn = keras.layers.BatchNormalization()
        
        # Handle activation
        if isinstance(activation, str):
            self.activation = keras.layers.Activation(activation)
        else:
            self.activation = activation
            
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.activation(x)
        return x


class DenseBlock(keras.layers.Layer):
    """
    Reusable dense block with Dense -> BatchNorm -> Activation -> Dropout
    """
    
    def __init__(self, units, activation='relu', dropout_rate=0.5, 
                 use_batch_norm=True, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        
        self.dense = keras.layers.Dense(units)
        self.use_batch_norm = use_batch_norm
        
        if use_batch_norm:
            self.bn = keras.layers.BatchNormalization()
            
        # Handle activation
        if isinstance(activation, str):
            self.activation = keras.layers.Activation(activation)
        else:
            self.activation = activation
            
        self.dropout = keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=False):
        x = self.dense(inputs)
        
        if self.use_batch_norm:
            x = self.bn(x, training=training)
            
        x = self.activation(x)
        x = self.dropout(x, training=training)
        return x