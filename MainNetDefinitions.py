import torch
import torch.nn as nn
import random
import numpy as np
from torchmetrics import PearsonCorrCoef
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
IMGSIZE = 256

class HazelNet(nn.Module):
    """Class for instanciating the siamese NN

    Args:
        nn (nn.Module): super class to inherit from in pytorch
    """

    def __init__(self, blocks, kernel_size, stride, padding, seed, modelname="dummy"):
        """Constructor for initializing siamese nn

        Args:
            blocks (int): number of cnn blocks to use
            kernel_size (int): size of sliding kernel, always square
            stride (int): stride with which kernel is applied
            padding (int): padding applied in convolutions
            seed (int): random state
            modelname (str, optional): name used for model . Defaults to "dummy".
        """
        
        torch.clear_autocast_cache(),  torch.cuda.empty_cache(), torch.manual_seed(seed), torch.cuda.manual_seed(seed), random.seed(seed), np.random.seed(seed),torch.manual_seed(seed)
        super(HazelNet, self).__init__()
        self.transfernet = basecnn(blocks=blocks, kernel_size=kernel_size, stride=stride, padding=padding)
        self.modelname = modelname   

        # add linear layers to compare between extracted features of the two images
        self.fc = nn.Sequential(
            nn.LazyLinear(IMGSIZE),
            )        
    
    def forward_once(self, inputs):
        """Helper function for forward path

        Args:
            inputs (tensor): input tensor

        Returns:
            tensor: output tensor
        """
        output = self.transfernet(inputs)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output
    
    def distance_layer(self, vec1, vec2):
        """Function for calculating the similarity between two tensors

        Args:
            vec1 (tensor): tensor for template images
            vec2 (tensor): tensor for images to compare with

        Returns:
            tensor: tensor containing the calculated similarity as float
        """
        num_outputs = vec1.shape[0]
        distancefunction =  PearsonCorrCoef(num_outputs=num_outputs).to(device)
        similarity = distancefunction(vec1.transpose(0,1), vec2.transpose(0,1)) if num_outputs > 1 else distancefunction(vec1.transpose(0,1)[:,0], vec2.transpose(0,1)[:,0]) # fix for bug in pearson corr coeff, see https://github.com/Lightning-AI/torchmetrics/issues/1647
        return similarity

    def forward(self, template, img):
        """Main function for forward path

        Args:
            template (tensor): tensor of template images
            img (tensor): tensor of images to compare

        Returns:
            tensor: tensor containing the calculated similarity as float
        """
        output1 = self.forward_once(template)
        output2 = self.forward_once(img)
        output = self.distance_layer(output1,output2)
 
        return output     

    
class basecnn(nn.Module):
    """Class for instanciating the NN for feature extraction

    Args:
        nn (nn.Module): super class to inherit from in pytorch
    """

    def __init__(self, blocks, kernel_size, stride, padding):
        """Constructor for initializing feature extraction network

        Args:
            blocks (int): number of cnn blocks to use
            kernel_size (int): size of sliding kernel, always square
            stride (int): stride with which kernel is applied
            padding (int): padding applied in convolutions
        """
        super(basecnn, self).__init__()
        modulelist = nn.ModuleList()
        for i in range(0,blocks):
            postchannels = int(256 / 2**(i))
            prechannels = int(256 / 2**(i+1)) if (i + 1) < blocks else 1
            block = nn.Sequential(
                nn.Conv2d(prechannels, postchannels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(postchannels),
                nn.Dropout2d(0.5),
                )
            modulelist = nn.ModuleList().append(block) + modulelist
        modulelist.append(nn.Flatten())
        self.imgLayer = nn.Sequential(*modulelist)

    def forward(self,img):
        """Main function for applyiny feature extraction

        Args:
            img (tensor): img used for feature extraction

        Returns:
            tensor: img passed through network
        """
        output = self.imgLayer(img)
        return  output