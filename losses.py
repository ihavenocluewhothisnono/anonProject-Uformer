import torch
import torch.nn as nn
import torch.nn.functional as F




#NEW EXTENSION


import torchvision.models as models


class VGGFeatureExtractor(nn.Module):
   def __init__(self, requires_grad=False):
       super(VGGFeatureExtractor, self).__init__()
       vgg16 = models.vgg16(pretrained=True)
       self.features = nn.Sequential(*list(vgg16.features.children())[:16]).eval()
       if not requires_grad:
           for param in self.features.parameters():
               param.requires_grad = False


   def forward(self, x):
       return self.features(x)


vgg_extractor = VGGFeatureExtractor().cuda()
perceptual_criterion = nn.MSELoss()


def perceptual_loss(pred, target):
    if pred.shape[1] == 1:
        pred = pred.repeat(1, 3, 1, 1)   # [B, 1, H, W] → [B, 3, H, W]
    if target.shape[1] == 1:
        target = target.repeat(1, 3, 1, 1)
    pred_features = vgg_extractor(pred)
    target_features = vgg_extractor(target)
    return perceptual_criterion(pred_features, target_features)




#end


def tv_loss(x, beta = 0.5, reg_coeff = 5):
   '''Calculates TV loss for an image `x`.
      
   Args:
       x: image, torch.Variable of torch.Tensor
       beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
   '''
   dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
   dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
   a,b,c,d=x.shape
   return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))


class TVLoss(nn.Module):
   def __init__(self, tv_loss_weight=1):
       super(TVLoss, self).__init__()
       self.tv_loss_weight = tv_loss_weight


   def forward(self, x):
       batch_size = x.size()[0]
       h_x = x.size()[2]
       w_x = x.size()[3]
       count_h = self.tensor_size(x[:, :, 1:, :])
       count_w = self.tensor_size(x[:, :, :, 1:])
       h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
       w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
       return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


   @staticmethod
   def tensor_size(t):
       return t.size()[1] * t.size()[2] * t.size()[3]




class CharbonnierLoss(nn.Module):
   """Charbonnier Loss (L1)"""


   def __init__(self, eps=1e-3):
       super(CharbonnierLoss, self).__init__()
       self.eps = eps


   def forward(self, x, y):
       diff = x - y
       # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
       loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
       return loss