"""
ResNet implementation
"""
import tensorflow as tf
from tensorflow import keras
from models.base_model import BaseModel
from activation_functions.activations import get_activation


class ResNet(BaseModel):
    """
    ResNet architecture with residual connections
    Original paper: Deep Residual Learning for Image Recognition
    """
    
    def __init__(self, input_shape, num_classes, activation='relu', depth=18):
        super().__init__(input_shape, num_classes, activation)
        self.depth = depth  # 18, 34, 50
        
    def _residual_block(self, x, filters, strides=1):
        """Basic residual block"""
        act_fn = get_activation(self.activation)
        
        shortcut = x
        
        # First conv
        x = keras.layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # Second conv
        x = keras.layers.Conv2D(filters, 3, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Adjust shortcut if dimensions changed
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(filters, 1, strides=strides)(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        
        # Add shortcut and apply activation
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation(act_fn)(x)
        
        return x
    
    def _bottleneck_block(self, x, filters, strides=1):
        """Bottleneck residual block for deeper networks"""
        act_fn = get_activation(self.activation)
        
        shortcut = x
        
        # 1x1 conv - reduce dimension
        x = keras.layers.Conv2D(filters, 1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # 3x3 conv
        x = keras.layers.Conv2D(filters, 3, strides=strides, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # 1x1 conv - restore dimension
        x = keras.layers.Conv2D(filters * 4, 1, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        
        # Adjust shortcut
        if strides != 1 or shortcut.shape[-1] != filters * 4:
            shortcut = keras.layers.Conv2D(filters * 4, 1, strides=strides)(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)
        
        x = keras.layers.Add()([x, shortcut])
        x = keras.layers.Activation(act_fn)(x)
        
        return x
    
    def _make_layer(self, x, filters, blocks, strides=1, use_bottleneck=False):
        """Create a layer with multiple residual blocks"""
        block_fn = self._bottleneck_block if use_bottleneck else self._residual_block
        
        x = block_fn(x, filters, strides)
        for _ in range(1, blocks):
            x = block_fn(x, filters, strides=1)
        
        return x
        
    def build_model(self):
        """Build ResNet model"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial conv
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs
            
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        act_fn = get_activation(self.activation)
        
        # Stem
        x = keras.layers.Conv2D(64, 7, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)
        
        # ResNet blocks based on depth
        if self.depth == 18:
            x = self._make_layer(x, 64, 2)
            x = self._make_layer(x, 128, 2, strides=2)
            x = self._make_layer(x, 256, 2, strides=2)
            x = self._make_layer(x, 512, 2, strides=2)
        elif self.depth == 34:
            x = self._make_layer(x, 64, 3)
            x = self._make_layer(x, 128, 4, strides=2)
            x = self._make_layer(x, 256, 6, strides=2)
            x = self._make_layer(x, 512, 3, strides=2)
        elif self.depth == 50:
            x = self._make_layer(x, 64, 3, use_bottleneck=True)
            x = self._make_layer(x, 128, 4, strides=2, use_bottleneck=True)
            x = self._make_layer(x, 256, 6, strides=2, use_bottleneck=True)
            x = self._make_layer(x, 512, 3, strides=2, use_bottleneck=True)
        
        # Global average pooling and classifier
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=f'ResNet{self.depth}')
        return self.model


class ResNet18(ResNet):
    """ResNet-18"""
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation, depth=18)


class ResNet34(ResNet):
    """ResNet-34"""
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation, depth=34)


class ResNet50(ResNet):
    """ResNet-50"""
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation, depth=50)