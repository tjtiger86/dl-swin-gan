import torch
import numpy as np
from torchvision.models import vgg16
from torchvision.models.feature_extraction import create_feature_extractor

class VGG_Loss(torch.nn.Module):
    def __init__(self, resize=True, feature_layers=[4, 9, 16]):
        super(VGG_Loss, self).__init__()
        model = vgg16(pretrained=True)
        #return_nodes = {'features.1': 'features1'}
        self.feature_layers = feature_layers
        self.features = create_feature_extractor(model, return_nodes={f'features.{k}': f'features{k}'
                                                                        for k in feature_layers})
        
        #self.features = create_feature_extractor(model, return_nodes)
        
        """
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        """
        
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, style_layers=[]):
    #def forward(self, input, target, feature_layers=[1, 2, 4, 8], style_layers=[]):
        if input.shape[1] != 3:
            #print(input.shape)
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0

        x = input
        y = target

        x = self.features(x)
        with torch.no_grad():
            y = self.features(y)

        W = torch.tensor(np.array([0.65, 0.3, 0.05]))
        
        for i, j in enumerate(self.feature_layers):
            loss += W[i]*torch.nn.functional.l1_loss(x.get(f'features{j}'), y.get(f'features{j}'))

        return loss