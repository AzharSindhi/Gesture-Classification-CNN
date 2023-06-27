from torchvision import models
import torch
from torch import nn
import torch.nn.functional as F


class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 512)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size//2)
        self.relu1 = nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size)
        self.relu2 = nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(hidden_size, out_size)
        self.relu3 = nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(out_size)



    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.bn1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        return out


class CustomModel(torch.nn.Module):
    def __init__(self, num_classes, input_channels=3, include_raw_keypoints=True, include_distances=False, include_context=True, include_skelton=True):
        super().__init__()
        self.num_classes = num_classes
        self.include_raw_keypoints = include_raw_keypoints
        self.include_distances = include_distances
        self.include_context = include_context
        self.include_skelton = include_skelton
        self.use_branch = include_raw_keypoints or include_distances
        self.input_channels = input_channels
        self.fcnn_input_size = 21*include_raw_keypoints  + 21*include_distances # combined linear inputs
        self.fcnn_hidden_size = 512
        self.fcnn_outsize = 256
        self.fcnn_out_dim  = 256   + 256*include_context + 256*self.use_branch
        # skelton is added along extra dimensions
        self.crop_resnet_c = input_channels + 3*include_skelton
        
        self.fcnn = NeuralNet(
            self.fcnn_input_size, self.fcnn_hidden_size, self.fcnn_outsize
        ) if self.use_branch else None
        self.resnet_crop = self.get_fintetuned_resnet34(
            self.crop_resnet_c, use_pretrained=True)
        self.resnet_context = self.get_fintetuned_resnet34(
            input_channels, use_pretrained=True) if self.include_context else None
        self.resnet_skelton = self.get_fintetuned_resnet34(
            input_channels, use_pretrained=True) if self.include_skelton else None
        # self.resnet2 = self.get_fintetuned_resnet18(in_channels, use_pretrained=True)
        # self.device = "cuda:0"
        self.fc1 = torch.nn.Linear(self.fcnn_out_dim, self.fcnn_out_dim // 4)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(self.fcnn_out_dim // 4)
        
        self.fc2 = torch.nn.Linear(self.fcnn_out_dim // 4, self.fcnn_out_dim // 2)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(self.fcnn_out_dim // 2)
        
        self.fc3 = torch.nn.Linear(self.fcnn_out_dim // 2, self.fcnn_out_dim)
        self.relu3 = torch.nn.ReLU()
        self.bn3 = torch.nn.BatchNorm1d(self.fcnn_out_dim)
        
        self.fc_out = torch.nn.Linear(self.fcnn_out_dim, self.num_classes)

    def forward(self, img_in, linear_in):
        if self.include_skelton:
            crop_tensor = img_in[:, :6].detach().clone()
            channels_used = 6
        else:
            crop_tensor = img_in[:, :3].detach().clone()
            channels_used = 3

        x = self.resnet_crop(crop_tensor)
        # if self.include_skelton:
        #     skelton_tensor = img_in[:, channels_used:channels_used + 3].detach().clone()
        #     x_skelton = self.resnet_skelton(skelton_tensor)
        #     x = torch.concat((x, x_skelton), dim=1)
        #     channels_used +=3
        if self.include_context:
            context_tensor = img_in[:, channels_used:channels_used + 3].detach().clone()
            x_context = self.resnet_context(context_tensor)
            x = torch.concat((x, x_context), dim=1)
        if self.use_branch:
            linear_tensor = linear_in[:, :self.fcnn_input_size].detach().clone()
            x_linear = self.fcnn(linear_tensor)
            x = torch.concat((x, x_linear), dim=1)
        
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.dropout(x, training=self.training)
    
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(x, training=self.training)

        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.dropout(x, training=self.training)
        
        x = self.fc_out(x)
        # x = F.sigmoid(x)
        
        # x = F.log_softmax(x, dim=-1)

        return x

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_fintetuned_resnet18(self, in_channels, use_pretrained=True):
        model = models.resnet18(pretrained=use_pretrained)
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # take model upto fully connected layer
        # self.set_parameter_requires_grad(model, False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, self.fcnn_outsize)

        return model

    def get_fintetuned_resnet34(self, in_channels, use_pretrained=True):
        model = models.resnet101(use_pretrained)
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # take model upto fully connected layer
        # self.set_parameter_requires_grad(model, False)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, self.fcnn_outsize)

        return model
