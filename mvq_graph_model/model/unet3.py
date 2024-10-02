import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange
from lightning.pytorch import LightningModule
from loguru import logger
from torchvision.models.video import R3D_18_Weights


class Custom3DUNet(LightningModule):
    """Unet-style CNN with Resnet encoder."""

    def __init__(
        self,
        input_channels,
        pre_trained=True,
        output_channels=(256, 128, 64, 32),
        num_classes=1,
        activation="spatial",
        upsampling_type="trilinear",
        deep_supervision=True,
        t_pad=16,
        us_features=False,
        context=5.0,
    ):
        super().__init__()
        if us_features:
            # 'us_features' passes on the input image + 'focused' version of it
            num_classes = num_classes - 2

        self.pre_trained = pre_trained

        assert num_classes > 0
        self.us_features = us_features

        self.num_classes = num_classes
        self.activation = activation
        self.deep_supervision = deep_supervision
        self.t_pad = t_pad
        self.context = context

        if t_pad > 0:
            self.pad = nn.ReplicationPad3d(t_pad)
        else:
            self.pad = None

        if upsampling_type == "linear":
            upsampling_type = "trilinear"

        # Load pretrained 3D ResNet and adjust the first conv layer
        if self.pre_trained:
            logger.info("Using pre-trained network weights.")
            weights = R3D_18_Weights.KINETICS400_V1
        else:
            logger.info("Not using pre-trained network.")
            weights = None

        self.resnet3d = models.video.r3d_18(weights=weights)
        self.resnet3d.stem[0] = nn.Conv3d(
            input_channels,
            64,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False,
        )

        # Decoder layers with upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode=upsampling_type)
        self.decoder1 = self._make_decoder_layer(512 + 256, output_channels[0])
        self.decoder2 = self._make_decoder_layer(
            output_channels[0] + 128, output_channels[1]
        )
        self.decoder3 = self._make_decoder_layer(
            output_channels[1] + 64, output_channels[2]
        )
        self.decoder4 = self._make_decoder_layer(output_channels[2], output_channels[3])

        self.side_convs = nn.ModuleList()
        if self.deep_supervision:
            for output_ch in output_channels[:-1]:
                self.side_convs.append(nn.Conv3d(output_ch, num_classes, kernel_size=1))

        self.final_side_conv = nn.Conv3d(
            output_channels[-1], num_classes, kernel_size=1
        )

        # Final segmentation layer
        self.final_conv = nn.Conv3d(output_channels[-1], num_classes, kernel_size=1)

        # Activation Function
        self.activation_fn = None
        if self.activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif self.activation == "softmax":
            self.activation_fn = nn.Softmax(dim=1)
        elif self.activation == "spatial":
            self.activation_fn = nn.Softmax2d()

        # Adjusting skip connections
        self.skip_conv1 = nn.Conv3d(256, output_channels[0], kernel_size=1)
        self.skip_conv2 = nn.Conv3d(128, output_channels[1], kernel_size=1)
        self.skip_conv3 = nn.Conv3d(64, output_channels[2], kernel_size=1)

        self.scaling = nn.Parameter(
            torch.FloatTensor(
                [0.0] * num_classes,
            ).reshape(1, -1, 1, 1, 1),
            requires_grad=True,
        )

    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        result = {}

        if self.us_features:
            # Passing input directly on output:
            result["us_features"] = x

        if self.pad is not None:
            x_padded = self.pad(x)
        else:
            x_padded = x

        original_size = x_padded.size()

        # Encoder
        x = self.resnet3d.stem(x_padded)
        skip_connections = []
        for i in range(1, 5):
            layer = getattr(self.resnet3d, f"layer{i}")
            x = layer(x)
            if i < 4:  # Collect skip connections for first 3 layers
                skip_connections.append(x)

        side_outputs = []
        # Decoder with adjusted skip connections

        x = self.upsample(x)
        x = torch.cat([x, self.skip_conv1(skip_connections[-1])], dim=1)
        x = self.decoder1(x)
        if self.deep_supervision:
            side_outputs.append(self.side_convs[0](x))

        x = self.upsample(x)
        x = torch.cat([x, self.skip_conv2(skip_connections[-2])], dim=1)
        x = self.decoder2(x)
        if self.deep_supervision:
            side_outputs.append(self.side_convs[1](x))

        x = self.upsample(x)
        x = torch.cat([x, self.skip_conv3(skip_connections[-3])], dim=1)
        x = self.decoder3(x)
        if self.deep_supervision:
            side_outputs.append(self.side_convs[2](x))

        x = self.upsample(x)
        x = self.decoder4(x)
        final_side_output = self.final_side_conv(x)

        side_outputs.append(final_side_output)

        side_output_upsampled = [
            nn.functional.interpolate(
                side_output, size=original_size[2:5], mode=self.upsample.mode
            )
            for side_output in side_outputs
        ]

        x = torch.mean(torch.stack(side_output_upsampled, dim=0), dim=0)

        # Remove padding from the output to match original size

        if self.t_pad > 0:
            pad_slice = slice(self.t_pad, -self.t_pad)
            x = x[:, :, pad_slice, pad_slice, pad_slice]

        x = torch.exp(self.scaling) * x

        if self.activation_fn is not None:
            if self.activation == "spatial":
                n, c, d, h, w = x.size()  # Save the original size of the tensor
                x = rearrange(x, "n c d h w -> (n d) c h w")
                x = self.activation_fn(x)
                x = rearrange(x, "(n d) c h w -> n c d h w", n=n, d=d, c=c, h=h, w=w)
            else:
                # For other activations, apply directly
                x = self.activation_fn(x)

        result |= {"cnn_prediction": x[:, 0:2], "cnn_features": x[:, 2:]}

        if self.us_features:
            # Create 'focused' version, using distance prediction (first channel):
            dmap_pred = (result["cnn_prediction"][:, 0:1]).detach()
            scale = dmap_pred.shape[-1]
            dmap_pred = dmap_pred - dmap_pred.min()
            weight = torch.exp(-torch.pow(dmap_pred * scale / self.context, 2.0))

            result["focused_us_features"] = result["us_features"] * weight

        return result
