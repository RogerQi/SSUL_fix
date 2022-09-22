from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbone import resnet
from .backbone import resnet_renorm
from .backbone import mobilenetv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class pixel_classifier(nn.Module):
    def __init__(self, in_channel, num_classes):
        super().__init__()
        num_classes = np.sum(num_classes)
        self.num_classes = num_classes
        print(num_classes)
        print(self.num_classes)
        self.class_mat = nn.Conv2d(in_channel, self.num_classes, 1, bias = False)
        self.scale_factor = 10
    
    def forward(self, x, scale_factor=None):
        '''
        x: (B, in_channel, H, W)
        '''
        # x_norm: (B, in_channel, H, W) where x_norm[i, :, H, W] is the norm of
        # x[i, :, H, W]. That is, x/x_norm yields normalized value along channel axis
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5) # avoid div by zero
        class_mat_norm = torch.norm(self.class_mat.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.class_mat.weight.data)
        self.class_mat.weight.data = self.class_mat.weight.data.div(class_mat_norm + 1e-5)
        cos_dist = self.class_mat(x_normalized)
        if scale_factor is not None:
            return scale_factor * cos_dist
        else:
            return self.scale_factor * cos_dist

def _segm_resnet_renorm(name, backbone_name, num_classes, pretrained_backbone, bn_freeze):
    assert name == 'deeplabv3' and backbone_name == 'deeplabv3_resnet101_renorm'

    backbone = resnet_renorm.__dict__[backbone_name](pretrained=pretrained_backbone)
    classifier = pixel_classifier(256, num_classes)

    model = DeepLabV3(backbone, classifier, bn_freeze)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, bn_freeze):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier, bn_freeze)
    return model

def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone, bn_freeze):
    if output_stride==8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    if name=='deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier, bn_freeze)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, bn_freeze):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, 
                                pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)
    elif backbone=='resnet101_renorm' and arch_type=='deeplabv3':
        assert output_stride is None
        model = _segm_resnet_renorm('deeplabv3', 'deeplabv3_resnet101_renorm', num_classes, pretrained_backbone, bn_freeze)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, 
                             pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)
    else:
        raise NotImplementedError
        
    return model


# Deeplab v3

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)

def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)

def deeplabv3_resnet101_renorm(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101_renorm', num_classes, output_stride=None, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)

def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)


# Deeplab v3+

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, bn_freeze=False):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, bn_freeze=bn_freeze)


