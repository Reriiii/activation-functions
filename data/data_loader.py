"""
Data loader module for various datasets
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np


class DataLoader:
    """Load and preprocess datasets"""
    
    def __init__(self, dataset_name='mnist', batch_size=32, validation_split=0.1):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.num_classes = 10
        self.input_shape = None
        
    def load_data(self):
        """Load dataset based on dataset_name"""
        if self.dataset_name == 'mnist':
            return self._load_mnist()
        elif self.dataset_name == 'fashion_mnist':
            return self._load_fashion_mnist()
        elif self.dataset_name == 'cifar10':
            return self._load_cifar10()
        elif self.dataset_name == 'cifar100':
            self.num_classes = 100
            return self._load_cifar100()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def _load_mnist(self):
        """Load MNIST dataset"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape to (batch, height, width, channels)
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        self.input_shape = (28, 28, 1)
        
        # Convert labels to one-hot encoding
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        return self._create_datasets(x_train, y_train, x_test, y_test)
    
    def _load_fashion_mnist(self):
        """Load Fashion-MNIST dataset"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        
        self.input_shape = (28, 28, 1)
        
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        return self._create_datasets(x_train, y_train, x_test, y_test)
    
    def _load_cifar10(self):
        """Load CIFAR-10 dataset"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        self.input_shape = (32, 32, 3)
        
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        return self._create_datasets(x_train, y_train, x_test, y_test)
    
    def _load_cifar100(self):
        """Load CIFAR-100 dataset"""
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
        
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        self.input_shape = (32, 32, 3)
        
        y_train = keras.utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.to_categorical(y_test, self.num_classes)
        
        return self._create_datasets(x_train, y_train, x_test, y_test)
    
    def _create_datasets(self, x_train, y_train, x_test, y_test):
        """Create train, validation and test datasets"""
        # Split training data into train and validation
        val_samples = int(len(x_train) * self.validation_split)
        
        x_val = x_train[:val_samples]
        y_val = y_train[:val_samples]
        x_train = x_train[val_samples:]
        y_train = y_train[val_samples:]
        
        # Create tf.data.Dataset objects
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(10000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset
    
    def get_data_info(self):
        """Return dataset information"""
        return {
            'dataset_name': self.dataset_name,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'batch_size': self.batch_size
        }