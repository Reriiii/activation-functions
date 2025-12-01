"""
ResNet-V2 implementation
"""
import tensorflow as tf
from tensorflow import keras
from models.base_model import BaseModel
from activation_functions.activations import get_activation

class ResNetV2(BaseModel):
    """
    ResNet-v2 architecture (pre-activation)
    Paper: Identity Mappings in Deep Residual Networks (He et al., 2016)
    """

    def __init__(self, input_shape, num_classes, activation='relu', depth=18):
        super().__init__(input_shape, num_classes, activation)
        self.depth = depth

    def _residual_block_v2(self, x, filters, strides=1):
        """Basic pre-activation residual block (v2)"""
        act_fn = get_activation(self.activation)
        shortcut = x

        # Pre-activation
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)

        # Shortcut projection if dimension mismatch
        if strides != 1 or shortcut.shape[-1] != filters:
            shortcut = keras.layers.Conv2D(filters, 1, strides=strides)(x)

        # First conv
        x = keras.layers.Conv2D(filters, 3, strides=strides, padding='same')(x)

        # Pre-activation for second conv
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)

        # Second conv
        x = keras.layers.Conv2D(filters, 3, padding='same')(x)

        return keras.layers.Add()([x, shortcut])


    def _bottleneck_block_v2(self, x, filters, strides=1):
        """Bottleneck pre-activation block for ResNet-v2"""
        act_fn = get_activation(self.activation)
        shortcut = x

        # Pre-activation
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)

        # Shortcut projection
        if strides != 1 or shortcut.shape[-1] != filters * 4:
            shortcut = keras.layers.Conv2D(filters * 4, 1, strides=strides)(x)

        # 1x1 conv
        x = keras.layers.Conv2D(filters, 1, padding='same')(x)

        # 3x3 conv
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Conv2D(filters, 3, strides=strides, padding='same')(x)

        # 1x1 conv
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Conv2D(filters * 4, 1, padding='same')(x)

        return keras.layers.Add()([x, shortcut])


    def _make_layer(self, x, filters, blocks, strides=1, use_bottleneck=False):
        """Create multiple pre-activation blocks"""
        block_fn = self._bottleneck_block_v2 if use_bottleneck else self._residual_block_v2

        x = block_fn(x, filters, strides)
        for _ in range(1, blocks):
            x = block_fn(x, filters, 1)
        return x


    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)

        # Resize small images to 32x32
        if self.input_shape[0] < 32:
            x = keras.layers.Resizing(32, 32)(inputs)
        else:
            x = inputs

        # Expand grayscale to 3 channels
        if self.input_shape[-1] == 1:
            x = keras.layers.Conv2D(3, 1, padding='same')(x)

        # Stem (v2 does NOT use BN+ReLU before stem conv)
        x = keras.layers.Conv2D(64, 7, strides=2, padding='same')(x)
        x = keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

        # ResNet-V2 block configuration
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

        # Final BN + ReLU before classifier (important for v2!)
        x = keras.layers.BatchNormalization()(x)
        act_fn = get_activation(self.activation)
        x = keras.layers.Activation(act_fn)(x)

        # Global average pooling + classifier
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs, outputs, name=f'ResNetV2-{self.depth}')
        return self.model



# Concrete classes
class ResNet18V2(ResNetV2):
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation, depth=18)


class ResNet34V2(ResNetV2):
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation, depth=34)


class ResNet50V2(ResNetV2):
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation, depth=50)
