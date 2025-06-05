import torch.nn as nn

import torchvision


class Resnet50(nn.Module):
    def __init__(self, net):
        super(Resnet50, self).__init__()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stage1 = net.layer1
        self.stage2 = net.layer2
        self.stage3 = net.layer3
        self.stage4 = net.layer4
        self.stg = net.avgpool
        self.fc = net.fc


    def forward(self, imgs):
        feats = self.stem(imgs)
        conv1 = self.stage1(feats)  # 18, 34: 64
        conv2 = self.stage2(conv1)
        conv3 = self.stage3(conv2)
        conv4 = self.stage4(conv3)

        return [conv1, conv2, conv3, conv4]



def build_backbone(pretrained=True):
    cnn = getattr(torchvision.models, 'resnet50')(pretrained=pretrained)
    backbone = Resnet50(cnn)
    return backbone
