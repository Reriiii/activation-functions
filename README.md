# Activation Function Research Project

Nghiên cứu so sánh hiệu năng của các activation functions trên các neural network architectures khác nhau.

## 📋 Tổng quan

Project này cho phép bạn:
- So sánh nhiều activation functions (ReLU, Sigmoid, Tanh, Swish, GELU, Mish, v.v.)
- Thử nghiệm trên nhiều model architectures (AlexNet, VGG16, ResNet, EfficientNet)
- Đánh giá trên các datasets chuẩn (MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100)
- Tự động tạo visualizations và báo cáo chi tiết

## 🗂️ Cấu trúc Project

```
activation-function-research/
│
├── data/                       # Data loading
│   └── data_loader.py
│
├── models/                     # Model architectures
│   ├── base_model.py
│   ├── alexnet.py
│   ├── vgg16.py
│   ├── resnet.py
│   └── efficientnet.py
│
├── activation_functions/       # Activation functions
│   └── activations.py
│
├── training/                   # Training utilities
│   └── trainer.py
│
├── evaluation/                 # Evaluation & visualization
│   └── visualization.py
│
├── experiments/                # Experiment runner
│   ├── run_experiment.py
│   └── experiment_config.yaml
│
├── results/                    # Output directory
│   ├── models/                 # Saved models
│   ├── logs/                   # Training logs
│   └── plots/                  # Visualizations
│
├── notebooks/                  # Jupyter notebooks
│   └── analysis.ipynb
│
├── requirements.txt
├── main.py
└── README.md
```

## 🚀 Cài đặt

### 1. Clone repository hoặc tạo thư mục project

```bash
mkdir activation-function-research
cd activation-function-research
```

### 2. Tạo virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 💻 Sử dụng

### Quick Start - Test nhanh

```bash
# Test nhanh với 1 epoch
python main.py --quick-test
```

### Chạy tất cả experiments

```bash
# Chạy tất cả các thí nghiệm trong config
python main.py
```

### Chạy experiment cụ thể

```bash
# Chỉ định dataset, model và activation
python main.py --dataset mnist --model alexnet_small --activation relu
```

### Sử dụng custom config

```bash
# Dùng file config tùy chỉnh
python main.py --config my_experiment_config.yaml
```

### Tắt GPU

```bash
# Chạy trên CPU
python main.py --no-gpu
```

## ⚙️ Configuration

Chỉnh sửa file `experiments/experiment_config.yaml`:

```yaml
# Training parameters
epochs: 50
batch_size: 128
learning_rate: 0.001
optimizer: 'adam'

# Datasets
datasets:
  - 'mnist'
  - 'fashion_mnist'
  - 'cifar10'

# Models
models:
  - 'alexnet_small'
  - 'vgg16_small'
  - 'resnet18'
  - 'efficientnet_small'

# Activation functions
activations:
  - 'relu'
  - 'sigmoid'
  - 'tanh'
  - 'swish'
  - 'gelu'
  - 'mish'
```

## 🎯 Thêm Activation Function mới

### 1. Thêm vào `activation_functions/activations.py`

```python
@tf.function
def my_new_activation(x):
    """
    Your custom activation function
    
    Example: f(x) = x * sigmoid(x) * tanh(x)
    """
    return x * tf.nn.sigmoid(x) * tf.nn.tanh(x)

# Thêm vào dictionary
ACTIVATION_FUNCTIONS = {
    # ... existing activations
    'my_new_activation': 'my_new_activation'
}

# Thêm vào get_activation function
def get_activation(name):
    # ...
    elif name == 'my_new_activation':
        return my_new_activation
```

### 2. Thêm vào config file

```yaml
activations:
  - 'relu'
  - 'my_new_activation'  # Add your new activation
```

### 3. Chạy experiments

```bash
python main.py --activation my_new_activation --dataset mnist --model resnet18
```

## 📊 Kết quả và Visualizations

Sau khi chạy experiments, kết quả được lưu trong thư mục `results/`:

### Files được tạo:

1. **CSV Results**: `results/all_results.csv`
   - Bảng tổng hợp tất cả các metrics

2. **Model Checkpoints**: `results/models/`
   - Best models cho mỗi experiment

3. **Training Logs**: `results/logs/`
   - TensorBoard logs
   - CSV training history

4. **Visualizations**: `results/plots/`
   - Training history plots
   - Comparison charts
   - Heatmaps
   - Comprehensive report

### Xem TensorBoard

```bash
tensorboard --logdir results/logs
```

## 📈 Metrics được tracking

- **Accuracy**: Test accuracy
- **Top-5 Accuracy**: Top-5 categorical accuracy
- **Loss**: Categorical cross-entropy loss
- **Precision & Recall**: Classification metrics
- **Training Time**: Thời gian training
- **Model Parameters**: Số lượng parameters

## 🔬 Các Models có sẵn

1. **AlexNet** (`alexnet`, `alexnet_small`)
   - Classic CNN architecture
   - Adapted cho small images

2. **VGG16** (`vgg16`, `vgg16_small`)
   - Deep architecture với 16 layers
   - Smaller version cho faster training

3. **ResNet** (`resnet18`, `resnet34`, `resnet50`)
   - Residual connections
   - Các độ sâu khác nhau

4. **EfficientNet** (`efficientnet_small`)
   - Modern efficient architecture
   - Mobile Inverted Bottleneck blocks

## 🎨 Activation Functions

### Standard Activations:
- **ReLU**: Rectified Linear Unit
- **Sigmoid**: Logistic function
- **Tanh**: Hyperbolic tangent
- **Softmax**: Multi-class output

### Advanced Activations:
- **Swish**: Self-gated activation
- **GELU**: Gaussian Error Linear Unit
- **Mish**: x * tanh(softplus(x))
- **ELU**: Exponential Linear Unit
- **SELU**: Scaled ELU

### Custom Activations:
- Thêm của riêng bạn!

## 📝 Ví dụ về kết quả

```
EXPERIMENT SUMMARY
================================================================================

1. Best Overall Performance:
   Experiment: mnist_resnet18_gelu
   Test Accuracy: 0.9934
   Model: resnet18
   Activation: gelu
   Dataset: mnist

2. Best Activation Function (Average):
   gelu: 0.9847
   swish: 0.9831
   mish: 0.9819

3. Best Model Architecture (Average):
   resnet18: 0.9856
   efficientnet_small: 0.9823
   vgg16_small: 0.9801
```

## 🛠️ Troubleshooting

### Out of Memory (OOM)

```bash
# Giảm batch size trong config
batch_size: 64  # thay vì 128

# Hoặc dùng smaller models
models:
  - 'alexnet_small'
  - 'vgg16_small'
```

### Slow Training

```bash
# Giảm số epochs
epochs: 20

# Dùng fewer combinations
datasets:
  - 'mnist'  # only one dataset
```

### GPU không được sử dụng

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📚 Tài liệu tham khảo

1. **AlexNet**: [ImageNet Classification with Deep CNNs](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

2. **VGG**: [Very Deep Convolutional Networks](https://arxiv.org/abs/1409.1556)

3. **ResNet**: [Deep Residual Learning](https://arxiv.org/abs/1512.03385)

4. **EfficientNet**: [EfficientNet: Rethinking Model Scaling](https://arxiv.org/abs/1905.11946)

5. **Swish**: [Searching for Activation Functions](https://arxiv.org/abs/1710.05941)

6. **GELU**: [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415)

7. **Mish**: [Mish: A Self Regularized Non-Monotonic Activation](https://arxiv.org/abs/1908.08681)

## 🤝 Contributing

Đóng góp của bạn luôn được hoan nghênh! Một số ý tưởng:

1. Thêm activation functions mới
2. Implement thêm model architectures
3. Thêm datasets mới
4. Cải thiện visualizations
5. Optimize training speed

## 📄 License

MIT License - feel free to use for research and education!

## 👨‍💻 Author

Your Name - Activation Function Research Project

## 🙏 Acknowledgments

- TensorFlow team
- Keras team
- OpenAI & Anthropic for AI assistance
- Research community for papers and implementations

---

**Happy Researching! 🚀**