import torch.nn as nn

# Model definitions: Discriminators
class Discriminator(nn.Module):
  def __init__(self, input_channels, num_filters=64):
    super(Discriminator, self).__init__()

    self.model = nn.Sequential(
        nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(num_filters*2),
        nn.LeakyReLU(0.2, inplace=True),

        # We can add more layers as we see fit

        nn.Conv2d(num_filters*2, 1, kernel_size=4, stride=1, padding=0) # Output a single scalar value
    )
  def forward(self, x):
    x = self.model(x)
    return x