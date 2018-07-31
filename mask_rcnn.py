"""
Mask R-CNN
The main Mask R-CNN model implemenetation.
"""
import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lib.nms_wrapper import nms
from torch.autograd import Variable
from lib.roi_align.crop_and_resize import CropAndResize
from tasks.bbox.generate_anchors import generate_pyramid_anchors


def to_variable(numpy_data, volatile=False):
    numpy_data = numpy_data.astype(np.float32)
    torch_data = torch.from_numpy(numpy_data).float()
    variable = Variable(torch_data, volatile=volatile)
    return variable


# ROIAlign function
def log2_graph(x):
    """Implementatin of Log2. pytorch doesn't have a native implemenation."""
    return torch.div(torch.log(x), math.log(2.))
    

def ROIAlign(feature_maps, rois, config, pool_size, mode='bilinear'):
    """Implements ROI Align on the features.

    Params:
    - pool_shape: [height, width] of the output pooled regions. Usually [7, 7]
    - image_shape: [height, width, chanells]. Shape of input image in pixels

    Inputs:
    - boxes: [batch, num_boxes, (x1, y1, x2, y2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, channels, height, width]

    Output:
    Pooled regions in the shape: [batch, num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    """
    [  x2-x1             x1 + x2 - W + 1  ]
    [  -----      0      ---------------  ]
    [  W - 1                  W - 1       ]
    [                                     ]
    [           y2-y1    y1 + y2 - H + 1  ]
    [    0      -----    ---------------  ]
    [           H - 1         H - 1      ]
    """

    # feature_maps= [P2, P3, P4, P5]
    rois = rois.detach()
    crop_resize = CropAndResize(pool_size, pool_size, 0)
    
    roi_number = rois.size()[1]
    
    pooled = rois.data.new(
            config.IMAGES_PER_GPU*rois.size(
            1), 256, pool_size, pool_size).zero_()
            
    rois = rois.view(
            config.IMAGES_PER_GPU*rois.size(1),
            4)
                   
    # Loop through levels and apply ROI pooling to each. P2 to P5.
    x_1 = rois[:, 0]
    y_1 = rois[:, 1]
    x_2 = rois[:, 2]
    y_2 = rois[:, 3]

    roi_level = log2_graph(
        torch.div(torch.sqrt((y_2 - y_1) * (x_2 - x_1)), 224.0))

    roi_level = torch.clamp(torch.clamp(
        torch.add(torch.round(roi_level), 4), min=2), max=5)

    # P2 is 256x256, P3 is 128x128, P4 is 64x64, P5 is 32x32
    # P2 is 4, P3 is 8, P4 is 16, P5 is 32
    for i, level in enumerate(range(2, 6)):

        scaling_ratio = 2**level

        height = float(config.IMAGE_MAX_DIM)/ scaling_ratio
        width = float(config.IMAGE_MAX_DIM) / scaling_ratio

        ixx = torch.eq(roi_level, level)

        box_indices = ixx.view(-1).int() * 0
        ix = torch.unsqueeze(ixx, 1)
        level_boxes = torch.masked_select(rois, ix)
        if len(level_boxes.size()) is 0:
            continue
        if level_boxes.size()[0] == 0:
            continue
        level_boxes = level_boxes.view(-1, 4)
        
        crops = crop_resize(feature_maps[i], torch.div(
                level_boxes, float(config.IMAGE_MAX_DIM)
                )[:, [1, 0, 3, 2]], box_indices)
                
        indices_pooled = ixx.nonzero()[:, 0]
        pooled[indices_pooled.data, :, :, :] = crops.data

    pooled = pooled.view(config.IMAGES_PER_GPU, roi_number,
               256, pool_size, pool_size)        
    pooled = Variable(pooled).cuda()
    return pooled
    
       
def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, :, 2] - boxes[:, :, 0]
    width = boxes[:, :, 3] - boxes[:, :, 1]
    center_y = boxes[:, :, 0] + 0.5 * height
    center_x = boxes[:, :, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, :, 0] * height
    center_x += deltas[:, :, 1] * width
    height *= torch.exp(deltas[:, :, 2])
    width *= torch.exp(deltas[:, :, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = [y1, x1, y2, x2]
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, 4] each row is y1, x1, y2, x2
    window: [4] in the form y1, x1, y2, x2
    """
    # Split corners
    wy1, wx1, wy2, wx2 = window
    y1, x1, y2, x2 = boxes
    # Clip

    y1 = torch.max(torch.min(y1, wy2), wy1)
    x1 = torch.max(torch.min(x1, wx2), wx1)
    y2 = torch.max(torch.min(y2, wy2), wy1)
    x2 = torch.max(torch.min(x2, wx2), wx1)

    clipped = torch.stack([x1, y1, x2, y2], dim=2)
    return clipped


# Backbone of the model
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=stride, bias=True)  # change
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001)
        if dilation == 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                                   padding=1, bias=True)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                                   padding=dilation, dilation=dilation, bias=True)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class resnet_graph(nn.Module):
    def __init__(self, block, layers, stage5=False):
        self.inplanes = 64
        super(resnet_graph, self).__init__()
        self.stage5 = stage5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(64, eps=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=0, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], dilation=2)

        if self.stage5:
            self.layer4 = self._make_layer(block, 512, layers[3], dilation=4)
            # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion,  eps=0.001),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # C1 has 64 channels
        C1 = self.maxpool(x)
        # C2 has 64x4 channels
        C2 = self.layer1(C1)
        # C3 has 128x4 channels
        C3 = self.layer2(C2)
        # C4 has 256x4 channels
        C4 = self.layer3(C3)
        # C5 has 512x4 channels
        if self.stage5:
            C5 = self.layer4(C4)
        else:
            C5 = None
        return C1, C2, C3, C4, C5


############################################################
#  Proposal Layer
############################################################
class rpn_graph(nn.Module):
    def __init__(self, input_dims, anchors_per_location,
                 anchor_stride):

        super(rpn_graph, self).__init__()
        # Setup layers
        self.rpn_conv_shared = nn.Conv2d(
            input_dims, 512, kernel_size=3, stride=anchor_stride, padding=1)
        self.rpn_class_raw = nn.Conv2d(
            512, 2 * anchors_per_location, kernel_size=1)
        self.rpn_bbox_pred = nn.Conv2d(
            512, 4 * anchors_per_location, kernel_size=1)

    def forward(self, x):
        shared = F.relu(self.rpn_conv_shared(x), True)
        x = self.rpn_class_raw(shared)
        rpn_class_logits = x.permute(
            0, 2, 3, 1).contiguous().view(x.size(0), -1, 2)          
        rpn_probs = F.softmax(rpn_class_logits, dim=-1)
        x = self.rpn_bbox_pred(shared)

        rpn_bbox = x.permute(0, 2, 3, 1).contiguous().view(
            x.size(0), -1, 4)  # reshape to (N, 4)

        return rpn_class_logits, rpn_probs, rpn_bbox


############################################################
#  Bbox Layer
#####################################m  #######################
# class fpn_classifier_graph(nn.Module):
#     def __init__(self, num_classes, config):
#         super(fpn_classifier_graph, self).__init__()
#         self.num_classes = num_classes
#         self.config = config
#
#         # Classifier head
#         self.mrcnn_class_logits = nn.Linear(1024, self.num_classes)
#
#     def forward(self, x):
#         mrcnn_probs = F.softmax(self.mrcnn_class_logits(x), dim=-1)
#
#         return mrcnn_probs

class fpn_classifier_graph(nn.Module):
    def __init__(self, num_classes, config):
        super(fpn_classifier_graph, self).__init__()
        self.num_classes = num_classes
        self.config = config
        # Setup layers
        self.mrcnn_class_conv1 = nn.Conv2d(
            256, 1024, kernel_size=self.config.POOL_SIZE, stride=1, padding=0)
        self.mrcnn_class_bn1 = nn.BatchNorm2d(1024, eps=0.001)

        #        self.dropout = nn.Dropout(p=0.5, inplace=True)

        self.mrcnn_class_conv2 = nn.Conv2d(
            1024, 1024, kernel_size=1, stride=1, padding=0)
        self.mrcnn_class_bn2 = nn.BatchNorm2d(1024, eps=0.001)

        # Classifier head
        self.mrcnn_class_logits = nn.Linear(1024, self.num_classes)
        self.mrcnn_bbox_fc = nn.Linear(1024, self.num_classes * 4)

    def forward(self, x, rpn_rois, classification=False):
        if classification:
            start = time.time()
            x = ROIAlign(x, rpn_rois, self.config, self.config.POOL_SIZE)
            spend = time.time() - start
            # print('first roalign', spend)
            roi_number = x.size()[1]
            x = x.view(self.config.IMAGES_PER_GPU * roi_number,
                       256, self.config.POOL_SIZE,
                       self.config.POOL_SIZE)
            x = self.mrcnn_class_conv1(x)
            x = self.mrcnn_class_bn1(x)
            x = F.relu(x, inplace=True)
            # x = self.dropout(x)
            x = self.mrcnn_class_conv2(x)
            x = self.mrcnn_class_bn2(x)
            x = F.relu(x, inplace=True)
            shared = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1)
            # Classifier head
            mrcnn_class_logits = self.mrcnn_class_logits(shared)
            mrcnn_probs = F.softmax(mrcnn_class_logits, dim=-1)
            x = self.mrcnn_bbox_fc(shared)
            mrcnn_bbox = x.view(x.size()[0], self.num_classes, 4)
            mrcnn_class_logits = mrcnn_class_logits.view(self.config.IMAGES_PER_GPU,
                                                         roi_number, mrcnn_class_logits.size()[-1])
            mrcnn_probs = mrcnn_probs.view(self.config.IMAGES_PER_GPU,
                                           roi_number, mrcnn_probs.size()[-1])
            # BBox head
            # [batch, boxes, num_classes , (dy, dx, log(dh), log(dw))]
            mrcnn_bbox = mrcnn_bbox.view(self.config.IMAGES_PER_GPU,
                                         roi_number, self.config.NUM_CLASSES, 4)
            return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

        else:
            mrcnn_probs = F.softmax(self.mrcnn_class_logits(x), dim=-1)
            return mrcnn_probs


############################################################
#  Main Class of MASK-RCNN
############################################################
class MaskRCNN(nn.Module):
    """
    Encapsulates the Mask RCNN model functionality.
    
    """
    def __init__(self, config, mode='inference', backbone_pretrain=False, body_pretrain=False):
        super(MaskRCNN, self).__init__()
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        self.config = config
        self.mode = mode
        self.resnet_graph = resnet_graph(
            Bottleneck, [3, 4, 23, 3], stage5=True)

        # feature pyramid layers:
        self.fpn_c5p5 = nn.Conv2d(
            512 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c4p4 = nn.Conv2d(
            256 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c3p3 = nn.Conv2d(
            128 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c2p2 = nn.Conv2d(
            64 * 4, 256, kernel_size=1, stride=1,  padding=0)

        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.scale_ratios = [4, 8, 16, 32]
        self.fpn_p6 = nn.MaxPool2d(
            kernel_size=1, stride=2, padding=0, ceil_mode=False)

        self.anchors = generate_pyramid_anchors(self.config.RPN_ANCHOR_SCALES,
                                                self.config.RPN_ANCHOR_RATIOS,
                                                self.config.BACKBONE_SHAPES,
                                                self.config.BACKBONE_STRIDES,
                                                self.config.RPN_ANCHOR_STRIDE)
        self.anchors = self.anchors.astype(np.float32)

        # RPN Model
        self.rpn = rpn_graph(256, len(self.config.RPN_ANCHOR_RATIOS),
                             self.config.RPN_ANCHOR_STRIDE)

        self.proposal_count = self.config.POST_NMS_ROIS_TRAINING if self.mode == "training"\
            else self.config.POST_NMS_ROIS_INFERENCE

        self.fpn_classifier_graph = fpn_classifier_graph(config.NUM_CLASSES, self.config)

    def forward(self, x):

        start = time.time()
        saved_for_loss = []
        C1, C2, C3, C4, C5 = self.resnet_graph(x)
        
        resnet_time = time.time()
       
        # print('resnet spend', resnet_time-start)
        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        P5 = self.fpn_c5p5(C5)

        # P4 = self.fpn_c4p4(C4) + F.upsample(P5,
        #                                    scale_factor=2, mode='bilinear')

        P4 = self.fpn_c4p4(C4) + P5

        P3 = self.fpn_c3p3(C3) + F.upsample(P4,
                                            scale_factor=2, mode='bilinear')

        # P3 = self.fpn_c3p3(C3) + P4

        P2 = self.fpn_c2p2(C2) + F.upsample(P3,
                                            scale_factor=2, mode='bilinear')
        # Attach 3x3 conv to all P layers to get the final feature maps.
        # P2 is 256, P3 is 128, P4 is 64, P5 is 32
        P2 = self.fpn_p2(P2)
        P3 = self.fpn_p3(P3)
        P4 = self.fpn_p4(P4)
        P5 = self.fpn_p5(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = self.fpn_p6(P5)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        self.mrcnn_feature_maps = [P2, P3, P4, P5]

        # Loop through pyramid layers
        rpn_class_logits_outputs = []
        rpn_class_outputs = []
        rpn_bbox_outputs = []

        for p in rpn_feature_maps:
            rpn_class_logits, rpn_probs, rpn_bbox = self.rpn(p)
            rpn_class_logits_outputs.append(rpn_class_logits)
            rpn_class_outputs.append(rpn_probs)
            rpn_bbox_outputs.append(rpn_bbox)

        rpn_class_logits = torch.cat(rpn_class_logits_outputs, dim=1)
        rpn_class = torch.cat(rpn_class_outputs, dim=1)
    
        rpn_bbox = torch.cat(rpn_bbox_outputs, dim=1)

        rpn_rois = self.proposal_layer(rpn_class, rpn_bbox)
 
        spend = time.time()-resnet_time

        return rpn_rois, self.mrcnn_feature_maps

    # bbox refinement including deltas apply, clip to border, NMS, etc.
    def proposal_layer(self, rpn_class, rpn_bbox):
        # handling proposals
        scores = rpn_class[:, :, 1]
        # Box deltas [batch, num_rois, 4]
        deltas_mul = Variable(torch.from_numpy(np.reshape(
            self.config.RPN_BBOX_STD_DEV, [1, 1, 4]).astype(np.float32))).cuda()
        deltas = rpn_bbox * deltas_mul

        pre_nms_limit = min(6000, self.anchors.shape[0])

        scores, ix = torch.topk(scores, pre_nms_limit, dim=-1,
                                largest=True, sorted=True)

        ix = torch.unsqueeze(ix, 2)
        ix = torch.cat([ix, ix, ix, ix], dim=2)
        deltas = torch.gather(deltas, 1, ix)

        _anchors = []
        for i in range(self.config.IMAGES_PER_GPU):
            anchors = Variable(torch.from_numpy(
                self.anchors.astype(np.float32))).cuda()
            _anchors.append(anchors)
        anchors = torch.stack(_anchors, 0) 
    
        pre_nms_anchors = torch.gather(anchors, 1, ix)
        refined_anchors = apply_box_deltas_graph(pre_nms_anchors, deltas)

        # Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
        size = self.config.IMAGE_MAX_DIM
        # height, width = self.config.IMAGE_SHAPE[:1]
        height, width = 1024, 1024
        window = np.array([0, 0, height, width]).astype(np.float32)
        window = Variable(torch.from_numpy(window)).cuda()
        refined_anchors_clipped = clip_boxes_graph(refined_anchors, window)

        refined_proposals = []
        for i in range(self.config.IMAGES_PER_GPU):
            indices = nms(
                torch.cat([refined_anchors_clipped.data[i], scores.data[i].view(6000, 1)], 1), 0.7)
            indices = indices[:self.proposal_count]
            indices = torch.stack([indices, indices, indices, indices], dim=1)
            indices = Variable(indices).cuda()
            proposals = torch.gather(refined_anchors_clipped[i], 0, indices)
            padding = self.proposal_count - proposals.size()[0]
            proposals = torch.cat(
                [proposals, Variable(torch.zeros([padding, 4])).cuda()], 0)
            refined_proposals.append(proposals)

        rpn_rois = torch.stack(refined_proposals, 0)

        return rpn_rois
            

