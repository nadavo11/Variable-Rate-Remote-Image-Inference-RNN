# VRIC-RNN
VARIABLE RATE IMAGE COMPRESSION WITH RNN - PYTORCH
## Features

- **Models**: Supports `fc`, `fc_rec`, `conv`, `conv_rec`, `lstm`.
- **Residual Learning**: Optional residual learning for supported models.
- **Customizable Training Parameters**: Fine-tune your training with various parameters.

## Installation

Provide instructions on how to set up the environment and install any dependencies. For example:

```bash
git clone https://yourrepo.git
cd yourproject
pip install -r requirements.txt
````
# usage:
 use the cli to train and evaluate models

## train 💪💪
python .\train.py

### example
```bash
python .\train.py --model conv --patch_size 32 --num_epochs 50 --batch_size 16 --log_step 3 --learning_rate 0.01 --coded_size 64
````

## Parameters

Detailed explanation of each parameter available in `train.py`:

- `--model `: Specifies the name of the model to be used. Options are:
  - `fc` (fully connected),
  - `fc_rec` (fully connected recursive),
  - `conv` (convolutional), 
  - `conv_rec` (convolutional recursive),
  - and `lstm` (long short-term memory).
`--residual` (type: `bool`, default: `False`): If set to `True`, enables residual connections in the model. Otherwise, the model does not use residual connections.

- `--batch_size` 
  - type: `int`
  - default: `4`
  - Defines the mini-batch size for training. A smaller batch size requires less memory but may affect model performance and training stability.

- `--coded_size` 
  - type: `int`
  - default: `4`:
  - The number of bits representing the encoded patch. This setting affects the level of detail in the encoded representation.

- `--patch_size` 
  - type: `int`
  - default: `8`:
  - Size of the encoded subdivisions of the input image. Larger sizes can capture more detail but increase computational complexity.

- `--num_passes` 
  - type: `int`
  - default: `16`
  - Specifies the number of passes for recursive architectures. More passes can potentially improve model accuracy at the cost of increased computation time.

- `--num_epochs` 
  - type: `int`
  - default: `3`
  - The total number of training epochs. Indicates how many times the entire dataset is processed by the model.

- `--learning_rate` 
  - type: `float`
  - default: `0.001`
  - Sets the learning rate for the optimizer, a crucial parameter affecting the speed and quality of training.

- `--weight_decay` 
  - type: `float`
  - default: `0`
  - Applies weight decay as a regularization technique to prevent overfitting by penalizing large weights.

- `--momentum` 
  - type: `float`
  - default: `0.9`
  - The momentum factor helps accelerate gradients vectors in the right directions, thus leading to faster converging.

- `--model_path` 
  - type: `str`
  - default: `./saved_models/`
  - The directory path where trained models are saved. Ensure this path exists or is writable.

- `--log_step` 
  - type: `int`
  - default: `10`
  - Frequency of logging training progress. A lower value means more frequent logging of training metrics.

- `--save_step` 
  - type: `int`
  - default: `5000`
  - Determines how often the model is saved to disk during training. Adjust based on your training duration and model complexity.
  Compression for low-resolution thumbnails images using RNN.

Original paper: Toderici et al., Google, ICLR 2016 
