"""
In this implementation:

We define a Generator class inheriting from nn.Module, which is the base class for all neural network modules in PyTorch.
The generator consists of an encoder-decoder architecture similar to the U-Net.
The encoder downsamples the input image to capture features at different scales.
The decoder upsamples the encoded features to generate the output image.
The final layer of the decoder uses the Tanh activation function to ensure that the output pixels are in the range [-1, 1].
"""

# Model definitions: Generators (First attempt architecture: U-Net)
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, input_channels, output_channels, num_filters=64):
    super(Generator, self).__init__()

    # Define the encoder part of U-Net
    self.encoder = nn.Sequential(
        nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(),

        nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=2),
        nn.BatchNorm2d(num_filters*2),
        nn.ReLU(),

        nn.Conv2d(num_filters*2, num_filters*4, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(num_filters*4),
        nn.ReLU(),

        # Add more layers as we see fit
    )

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(num_filters*4, num_filters*2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(num_filters*2),
        nn.ReLU(),

        nn.ConvTranspose2d(num_filters*2, num_filters, kernel_size=4, stride=1, padding=1),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(),

        nn.ConvTranspose2d(num_filters, output_channels, kernel_size=4, stride=2, padding=2),
        nn.Sigmoid() # Ensure the output is between 0 and 1
    )

  def forward(self, x):
    # Encoder
    encoded = self.encoder(x)

    # Decoder
    decoded = self.decoder(encoded)

    return decoded