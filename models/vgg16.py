"""
VGG16 implementation
"""
import tensorflow as tf
from tensorflow import keras
from models.base_model import BaseModel
from activation_functions.activations import get_activation


class VGG16(BaseModel):
    """
    VGG16 architecture
    Original paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
    """
    
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation)
        
    def _conv_block(self, x, filters, num_convs):
        """Helper function to create a VGG conv block"""
        act_fn = get_activation(self.activation)
        
        for _ in range(num_convs):
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(act_fn)(x)
        
        x = keras.layers.MaxPooling2D(2, strides=2)(x)
        return x
        
    def build_model(self):
        """Build VGG16 model"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Resize if input is too small
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs
            
        # Convert grayscale to RGB
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        # VGG16 blocks
        x = self._conv_block(x, 64, 2)   # Block 1
        x = self._conv_block(x, 128, 2)  # Block 2
        x = self._conv_block(x, 256, 3)  # Block 3
        x = self._conv_block(x, 512, 3)  # Block 4
        x = self._conv_block(x, 512, 3)  # Block 5
        
        # Classifier
        x = keras.layers.Flatten()(x)
        
        act_fn = get_activation(self.activation)
        
        x = keras.layers.Dense(4096)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        x = keras.layers.Dense(4096)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='VGG16')
        return self.model


class VGG16Small(BaseModel):
    """
    Smaller version of VGG16 for faster training
    """
    
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation)
        
    def _conv_block(self, x, filters, num_convs):
        """Helper function to create a VGG conv block"""
        act_fn = get_activation(self.activation)
        
        for _ in range(num_convs):
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(act_fn)(x)
        
        x = keras.layers.MaxPooling2D(2, strides=2)(x)
        return x
        
    def build_model(self):
        """Build smaller VGG16 model"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs
            
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        # Reduced filters
        x = self._conv_block(x, 32, 2)
        x = self._conv_block(x, 64, 2)
        x = self._conv_block(x, 128, 3)
        x = self._conv_block(x, 256, 3)
        
        x = keras.layers.Flatten()(x)
        
        act_fn = get_activation(self.activation)
        
        x = keras.layers.Dense(1024)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        x = keras.layers.Dense(512)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)
        
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='VGG16-Small')
        return self.model