import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# this encoder only supports resnet
class Encoder(nn.Module):
    def __init__(self, backbone_name:str):
        super(Encoder, self).__init__()
        # select a model
        if backbone_name == "resnet18":
            resnet = resnet18(pretrained=True)
        elif backbone_name == "resnet34":
            resnet = resnet34(pretrained=True)
        elif backbone_name == "resnet50":
            resnet = resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            resnet = resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            resnet = resnet152(pretrained=True)
        else:
            assert False, f"'{backbone_name}' backbone is not supported"
        
        # remove a fully connected layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # freeze all updatable weights of the encoder
        self._freeze_all(self.encoder)
    
    def _freeze_all(model:nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.encoder(x).squeeze()
        return x

# this convlstm only supports lstm
class ConvLSTM(nn.Module):
    def __init__(self, backbone_name:str, num_classes:int, hidden_size:int = 1024, num_layers:int = 1, bidirectional:bool = True):
        super(ConvLSTM, self).__init__()
        # freeze
        self.encoder = Encoder(backbone_name)
        # updateable
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # get shape
        b, d, c, h, w = x.shape

        # get (spatial)feature of frames
        x = x.view(b * d, c, h, w)
        x = self.encoder(x)

        # get (temporal)feature of frames
        x = x.view(b, d, -1)
        x = self.lstm(x)
        print(x.size())

        # get classifier scores
        x = self.classifier(x)

        return x