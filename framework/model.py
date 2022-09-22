import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, num_classes, dropout, rectify, maxpool, enc_sizes, n_in, n_out):
        super().__init__()

        def my_conv_block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1), rectify, maxpool
            )

        self.features1 = nn.Sequential(
            nn.Conv2d(n_in, 64, kernel_size=(7, 7), stride=1, padding=3),
            rectify,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.enc_sizes = [n_out, *enc_sizes]

        convolution_blocks = [
            my_conv_block(in_f, out_f)
            for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])
        ]

        self.features_blocks = nn.Sequential(*convolution_blocks)

        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            rectify,
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            rectify,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            rectify,
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            rectify,
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features1(x)
        x = self.features_blocks(x)
        x = self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
