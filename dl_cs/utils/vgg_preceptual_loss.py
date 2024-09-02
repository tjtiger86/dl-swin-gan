import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:6].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[6:9].eval())
        
        """
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        """

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[4, 5, 8], style_layers=[]):
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

        for i, block in enumerate(self.blocks):
            #Stupid hard coding the weights
            print("I is {}, features layres is {}".format(i, feature_layers))
            if i== 1:
                W = 0.65
            elif i == 4:
                W = 0.3
            elif i == 8:
                W = 0.05
            else:
                W = 0
            # The target does not require a gradient, but the input does. 
            
            x = block(x)
            with torch.no_grad():
                y = block(y)

            if i in feature_layers:
                print("I am in the feature layer")
                loss += W*torch.nn.functional.l1_loss(x, y)
            """
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
            """
        return loss
