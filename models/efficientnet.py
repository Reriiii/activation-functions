"""
EfficientNet implementation (simplified version)
"""
import tensorflow as tf
from tensorflow import keras
from models.base_model import BaseModel
from activation_functions.activations import get_activation


class MBConvBlock(keras.layers.Layer):
    """
    Mobile Inverted Residual Bottleneck Block
    Used in EfficientNet
    """
    
    def __init__(self, filters, kernel_size, strides, expand_ratio, activation='swish', **kwargs):
        super(MBConvBlock, self).__init__(**kwargs)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.activation_name = activation
        
    def build(self, input_shape):
        input_filters = input_shape[-1]
        expanded_filters = input_filters * self.expand_ratio
        
        act_fn = get_activation(self.activation_name)
        
        # Expansion phase
        if self.expand_ratio != 1:
            self.expand_conv = keras.layers.Conv2D(expanded_filters, 1, padding='same', use_bias=False)
            self.expand_bn = keras.layers.BatchNormalization()
            self.expand_act = keras.layers.Activation(act_fn)
        
        # Depthwise convolution
        self.dw_conv = keras.layers.DepthwiseConv2D(
            self.kernel_size,
            strides=self.strides,
            padding='same',
            use_bias=False
        )
        self.dw_bn = keras.layers.BatchNormalization()
        self.dw_act = keras.layers.Activation(act_fn)
        
        # Squeeze and Excitation
        se_filters = max(1, input_filters // 4)
        self.se_squeeze = keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.se_reduce = keras.layers.Conv2D(se_filters, 1, activation=act_fn)
        self.se_expand = keras.layers.Conv2D(expanded_filters, 1, activation='sigmoid')
        
        # Output phase
        self.project_conv = keras.layers.Conv2D(self.filters, 1, padding='same', use_bias=False)
        self.project_bn = keras.layers.BatchNormalization()
        
        # Skip connection
        self.use_skip = (self.strides == 1) and (input_shape[-1] == self.filters)
        
        super(MBConvBlock, self).build(input_shape)
    
    def call(self, inputs, training=False):
        x = inputs
        
        # Expansion
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x, training=training)
            x = self.expand_act(x)
        
        # Depthwise
        x = self.dw_conv(x)
        x = self.dw_bn(x, training=training)
        x = self.dw_act(x)
        
        # Squeeze and Excitation
        se = self.se_squeeze(x)
        se = self.se_reduce(se)
        se = self.se_expand(se)
        x = keras.layers.multiply([x, se])
        
        # Output
        x = self.project_conv(x)
        x = self.project_bn(x, training=training)
        
        # Skip connection
        if self.use_skip:
            x = keras.layers.add([x, inputs])
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "expand_ratio": self.expand_ratio,
            "activation": self.activation_name,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


class EfficientNet(BaseModel):
    """
    EfficientNet architecture (simplified B0 version)
    Original paper: EfficientNet: Rethinking Model Scaling for CNNs
    """
    
    def __init__(self, input_shape, num_classes, activation='swish'):
        super().__init__(input_shape, num_classes, activation)
        
    def build_model(self):
        """Build EfficientNet-B0 model"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        # Resize if needed
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs
            
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        act_fn = get_activation(self.activation)
        
        # Stem
        x = keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # MBConv blocks configuration: (filters, kernel_size, strides, expand_ratio, repeats)
        blocks_config = [
            (16, 3, 1, 1, 1),   # Stage 1
            (24, 3, 2, 6, 2),   # Stage 2
            (40, 5, 2, 6, 2),   # Stage 3
            (80, 3, 2, 6, 3),   # Stage 4
            (112, 5, 1, 6, 3),  # Stage 5
            (192, 5, 2, 6, 4),  # Stage 6
            (320, 3, 1, 6, 1),  # Stage 7
        ]
        
        for filters, kernel_size, strides, expand_ratio, repeats in blocks_config:
            for i in range(repeats):
                stride = strides if i == 0 else 1
                x = MBConvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    expand_ratio=expand_ratio,
                    activation=self.activation
                )(x)
        
        # Head
        x = keras.layers.Conv2D(1280, 1, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNet-B0')
        return self.model


class EfficientNetSmall(BaseModel):
    """
    Smaller/faster version of EfficientNet for quick experiments
    """
    
    def __init__(self, input_shape, num_classes, activation='swish'):
        super().__init__(input_shape, num_classes, activation)
        
    def build_model(self):
        """Build smaller EfficientNet model"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs
            
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        act_fn = get_activation(self.activation)
        
        # Stem
        x = keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # Simplified blocks
        blocks_config = [
            (16, 3, 1, 1, 1),
            (24, 3, 2, 6, 1),
            (40, 5, 2, 6, 1),
            (80, 3, 2, 6, 2),
            (160, 5, 1, 6, 1),
        ]
        
        for filters, kernel_size, strides, expand_ratio, repeats in blocks_config:
            for i in range(repeats):
                stride = strides if i == 0 else 1
                x = MBConvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    expand_ratio=expand_ratio,
                    activation=self.activation
                )(x)
        
        # Head
        x = keras.layers.Conv2D(640, 1, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNet-Small')
        return self.model

class EfficientNetB1(BaseModel):
    """
    EfficientNet-B1 architecture (slightly larger than B0)
    """
    
    def __init__(self, input_shape, num_classes, activation='swish'):
        super().__init__(input_shape, num_classes, activation)
        
    def build_model(self):
        """Build EfficientNet-B1"""
        
        inputs = keras.Input(shape=self.input_shape)
        
        # B1 uses 240x240 input
        resize_to = 240
        if self.input_shape[0] < resize_to:
            x = keras.layers.Resizing(resize_to, resize_to)(inputs)
        else:
            x = inputs
            
        # If grayscale, convert to RGB
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)
        
        act_fn = get_activation(self.activation)
        
        # Stem
        x = keras.layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        # EfficientNet-B1 blocks (same as B0)
        blocks_config = [
            (16, 3, 1, 1, 1),   # Stage 1
            (24, 3, 2, 6, 2),   # Stage 2
            (40, 5, 2, 6, 2),   # Stage 3
            (80, 3, 2, 6, 3),   # Stage 4
            (112, 5, 1, 6, 3),  # Stage 5
            (192, 5, 2, 6, 4),  # Stage 6
            (320, 3, 1, 6, 1),  # Stage 7
        ]
        
        for filters, kernel_size, strides, expand_ratio, repeats in blocks_config:
            for i in range(repeats):
                stride = strides if i == 0 else 1
                x = MBConvBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=stride,
                    expand_ratio=expand_ratio,
                    activation=self.activation
                )(x)
        
        # Head (B1 slightly bigger than B0)
        x = keras.layers.Conv2D(1280, 1, padding='same', use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)   # B1 uses dropout=0.3
        
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='EfficientNet-B1')
        return self.model
