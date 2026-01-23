"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models

class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x



class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        self.hp = hp
        ########################################################################
        #                             YOUR CODE                                #
        ########################################################################
        resnet = models.resnet18(pretrained=True)

        #ENCODER

        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        ) # size = 240*240*3 -> 60*60*64

        self.layer1 = resnet.layer1 # size = 60*60*64 -> 60*60*64
        self.layer2 = resnet.layer2 # size = 60*60*64 -> 30*30*128
        self.layer3 = resnet.layer3 # size = 30*30*128 -> 15*15*256

        #DECODER
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # size = 15*15*256 -> 30*30*128
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # size = 30*30*128 -> 60*60*64
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2) # size = 60*60*64 -> 120*120*64
        self.up4 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2) # size = 120*120*64 -> 240*240*num_classes

        pass
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        ########################################################################
        #                             YOUR CODE                                #  
        ########################################################################

        # ENCODER
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return x

    # @property
    # def is_cuda(self):
    #     """
    #     Check if model parameters are allocated on the GPU.
    #     """
    #     return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(nn.Module):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()

if __name__ == "__main__":
    from torchinfo import summary
    summary(SegmentationNN(), (1, 3, 240, 240), device="cpu")