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
    elif name == 'softsign_like':
        return softsign_like
    elif name == 'xtanh':
        return xtanh
    elif name == 'sinusoidal':
        return sinusoidal
    elif name == 'rsqrt_unit':
        return rsqrt_unit
    elif name == 'softplus_relu':
        return softplus_relu
    elif name == 'erf_act':
        return erf_act
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

# 1. Softsign-like activation: x / (1 + |x|)
@tf.function
def softsign_like(x):
    return x / (1.0 + tf.abs(x))


# 2. x * tanh(x): smooth, stable, non-monotonic like Mish/Swish
@tf.function
def xtanh(x):
    return x * tf.math.tanh(x)


# 3. Sinusoidal function: sin(x)
# Works like SIREN networks
@tf.function
def sinusoidal(x):
    return tf.sin(x)


# 4. x / sqrt(1 + x^2): smooth, bounded, stable gradient
@tf.function
def rsqrt_unit(x):
    return x / tf.sqrt(1.0 + x * x)


# 5. Softplus — smooth ReLU variant
@tf.function
def softplus_relu(x):
    return tf.nn.softplus(x)


# 6. erf(x) — smooth bell curve activation (GELU uses erf)
@tf.function
def erf_act(x):
    return tf.math.erf(x)

# 7. Snake Activation: x + sin(x)
# Khắc phục điểm yếu của Sinusoidal cũ.
# Nó thêm thành phần tuyến tính 'x' vào sin(x), giúp gradient không bao giờ bị triệt tiêu hoàn toàn.
# Rất mạnh cho việc học các đặc trưng có tính chu kỳ (như texture trong ảnh).
@tf.function
def snake_act(x):
    return x + tf.sin(x)

# 8. Bent Identity: ((sqrt(x^2 + 1) - 1) / 2) + x
# Hàm này "uốn cong" nhẹ quanh gốc tọa độ nhưng trở về tuyến tính khi x lớn.
# Cực kỳ an toàn cho các mạng sâu (Deep ResNet) vì nó hành xử giống Identity function ở khoảng giá trị lớn, tránh vanishing gradient.
@tf.function
def bent_identity(x):
    return ((tf.sqrt(x*x + 1.0) - 1.0) / 2.0) + x

# 9. Log-Linear: x * ln(1 + |x|)
# Một biến thể "nén" dữ liệu.
# Nó tăng trưởng chậm hơn ReLU (Logarithmic growth).
# Hữu ích khi dataset có nhiều ngoại lai (outliers) giá trị lớn gây nổ gradient, hàm này sẽ kìm hãm chúng lại.
@tf.function
def log_linear(x):
    return x * tf.math.log(1.0 + tf.abs(x))

# 10. Hard Mish: x * hard_tanh(softplus(x))
# Phiên bản xấp xỉ của Mish nhưng tính toán nhanh hơn do loại bỏ bớt hàm exp đắt đỏ.
# Sử dụng logic của MobileNetV3 để tối ưu tốc độ inference.
@tf.function
def hard_mish(x):
    # Softplus xấp xỉ nhanh
    sp = tf.nn.relu(x) + tf.math.log(1.0 + tf.exp(-tf.abs(x))) 
    # Hard Tanh xấp xỉ: clip giá trị trong khoảng [-1, 1]
    htanh = tf.clip_by_value(sp, -1.0, 1.0)
    return x * htanh

# 11. Penalized Tanh: tanh(x) - 0.2*x*x (chỉ ví dụ hệ số 0.2)
# Hoặc biến thể tốt hơn: Tanh-Shrink: x - tanh(x)
# Hàm này tập trung vào phần "dư" (residual) của tín hiệu. 
# Tại x gần 0, giá trị rất nhỏ (phẳng), giúp mạng thưa (sparse) hơn.
@tf.function
def tanh_shrink(x):
    return x - tf.math.tanh(x)

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
    'softsign_like': 'softsign_like',
    'xtanh': 'xtanh',
    'sinusoidal': 'sinusoidal',
    'rsqrt_unit': 'rsqrt_unit',
    'softplus_relu': 'softplus_relu',
    'erf_act': 'erf_act'
}