# MNIST WGAN-GP

This repository contains a script for training a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP) on the MNIST dataset. The WGAN-GP is an improvement over the traditional GANs, designed to address the issues of instability and mode collapse during training.

## Features

- **WGAN-GP Model**: Implements the WGAN with Gradient Penalty for stable training.
- **MNIST Dataset**: Trains the model on the MNIST dataset, which contains images of handwritten digits.
- **Image Generation**: Generates new handwritten digit images based on the trained model.

## Requirements

To run this script, you need to have the following Python packages installed:

- `torch`
- `torchvision`
- `numpy`
- `matplotlib`

You can install these packages using pip:

```bash
pip install torch torchvision numpy matplotlib
````

## Usage

1. Training the Model:
    - The script is designed to train a WGAN-GP model on the MNIST dataset. Training parameters such as the number of epochs, learning rate, and batch size can be adjusted within the script.

2. Generating Images:
    - After training, you can generate new images by running the script. The generated images will be saved in the ./figures/ directory as generated_output_003.png.

## Example

To train the model and generate images, simply run the notebook or script in your Python environment:

