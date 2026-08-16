import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Basic residual block.

    Input:
        [B, C, H, W]

    Output:
        [B, C, H, W]
    """

    def __init__(self, channels=64):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class RestorationNet(nn.Module):
    """
    KLA image restoration network.

    Input:
        NoisyLR: [B, 1, 128, 128]

    Output:
        Restored image: [B, 1, 256, 256]
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        channels=64,
        num_blocks=8,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Feature extraction
        # ---------------------------------------------------------

        self.head = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        # ---------------------------------------------------------
        # Residual feature processing
        # ---------------------------------------------------------

        self.body = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
        )

        # ---------------------------------------------------------
        # Upsampling
        #
        # 128x128 -> 256x256
        # ---------------------------------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * 4,
                kernel_size=3,
                padding=1,
            ),

            nn.PixelShuffle(2),

            nn.ReLU(inplace=True),
        )

        # ---------------------------------------------------------
        # Final image reconstruction
        # ---------------------------------------------------------

        self.tail = nn.Conv2d(
            channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        # ---------------------------------------------------------
        # Head
        # ---------------------------------------------------------

        features = self.head(x)

        # ---------------------------------------------------------
        # Residual body
        # ---------------------------------------------------------

        residual = self.body(features)

        residual = self.body_conv(residual)

        features = features + residual

        # ---------------------------------------------------------
        # Upscale 2x
        # ---------------------------------------------------------

        features = self.upsample(features)

        # ---------------------------------------------------------
        # Reconstruct grayscale image
        # ---------------------------------------------------------

        output = self.tail(features)

        # Ground truth is [0,1]
        output = torch.sigmoid(output)

        return output