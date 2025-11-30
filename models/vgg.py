import tensorflow as tf
from tensorflow import keras
from models.base_model import BaseModel
from activation_functions.activations import get_activation


class VGGBase(BaseModel):
    """
    Base class for VGG architectures (VGG11/13/16/19/Small)
    """

    def __init__(self, input_shape, num_classes, activation='relu', config=None):
        super().__init__(input_shape, num_classes, activation)
        self.config = config  # list defining how many conv layers per block

    def _conv_block(self, x, filters, num_convs):
        act_fn = get_activation(self.activation)

        for _ in range(num_convs):
            x = keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(act_fn)(x)

        x = keras.layers.MaxPooling2D(2)(x)
        return x

    def build_model(self):
        if self.config is None:
            raise ValueError("VGG config must be provided")

        inputs = keras.Input(shape=self.input_shape)

        # Input preprocessing
        x = inputs
        if self.input_shape[0] < 32:  # CIFAR-10 default: 32x32
            x = keras.layers.Resizing(32, 32)(x)
        if self.input_shape[-1] == 1:  # grayscale → RGB
            x = keras.layers.Conv2D(3, 1)(x)

        # Backbone
        filters = [64, 128, 256, 512, 512]
        for num_convs, f in zip(self.config, filters):
            x = self._conv_block(x, f, num_convs)

        # Classifier
        act_fn = get_activation(self.activation)

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(4096)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)

        x = keras.layers.Dense(4096)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(act_fn)(x)
        x = keras.layers.Dropout(0.5)(x)

        outputs = keras.layers.Dense(self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs)
        return self.model

class VGG11(VGGBase):
    """ VGG11 configuration (1-1-2-2-2 conv layers per block) """
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation,
                         config=[1, 1, 2, 2, 2])


class VGG13(VGGBase):
    """ VGG13 configuration """
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation,
                         config=[2, 2, 2, 2, 2])


class VGG16(VGGBase):
    """ VGG16 standard configuration """
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation,
                         config=[2, 2, 3, 3, 3])


class VGG19(VGGBase):
    """ VGG19 configuration """
    def __init__(self, input_shape, num_classes, activation='relu'):
        super().__init__(input_shape, num_classes, activation,
                         config=[2, 2, 4, 4, 4])