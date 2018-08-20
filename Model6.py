"""Model6 consists of two attention branch: entity-attention branch together with
a attribute-attention branch."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from config import Config
from mask_rcnn import MaskRCNN
import torch.nn.functional as F
from collections import OrderedDict
from CompactBilinearPooling import CompactBilinearPooling


class Model6(nn.Module):

    def __init__(self, opts, body_pretrain=False):
        super(Model6, self).__init__()

        # Load pre-trained back-boned model
        print('==> Building backbone model...')
        config = Config()
        config.IMAGES_PER_GPU = opts.batch_size
        config.NUM_CLASSES = opts.class_num

        # Load Back bone Module
        pretrained_weight = opts.backbone_model
        state_dict = torch.load(pretrained_weight)
        visual_net = MaskRCNN(config=config, mode='inference')
        model_keys = visual_net.state_dict().keys()
        for name, param in list(state_dict.items()):
            if name not in model_keys:
                del state_dict[name]

        # Load coco Pre-trained Res-net
        new_params = visual_net.state_dict()
        new_params.update(state_dict)
        visual_net.load_state_dict(new_params)
        visual_net.cuda()
        # visual_net.eval()
        print(visual_net.config)

        # Load Front bone module
        body = FrontBone(81)
        if body_pretrain:
            # for param in vnet.parameters():
            #     param.requires_grad = False
            # load pre trained  model
            path = os.path.join(os.getcwd(), 'checkpoint', body_pretrain)
            tnet = torch.load(path)
            new_state_dict = OrderedDict()
            for value in tnet['state_dict']:
                key = value.replace("module.", "")
                value = tnet['state_dict'][value]
                new_state_dict[key] = value
            body.load_state_dict(new_state_dict)

        # Load Attribute module
        attr_branch = AttributeBranch(75)

        # Freeze the back_bone model
        for param in visual_net.parameters():
            param.requires_grad = False

        # Freeze the classifier or Not
        for param in visual_net.fpn_classifier_graph.parameters():
            param.requires_grad = True

        # Freeze the entity branch
        for param in body.parameters():
            param.requires_grad = False

        self.attr_branch = attr_branch
        self.body = body
        self.opts = opts
        self.visual_net = visual_net

        # feature expanding to 1024 then feed to mask-RCNN classifier
        self.fc_p3 = nn.Linear(256, 1024)
        self.fc_p4 = nn.Linear(256, 1024)
        self.fc_p5 = nn.Linear(256, 1024)
        self.fc = nn.Linear(1024, 75)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.pool = nn.AvgPool2d(128)

    def retrieve_bboxes(self, map, bboxes):
        # compute the mean value of attention map
        mean = np.mean(map)
        # transform to 0-255 scale image
        test = (map * 255).astype('uint8')
        # threshold set to 3 times of mean value
        ret, thresh = cv2.threshold(test, round(mean * 3 * 255), 255, 0)
        # contour detection
        im2, cts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = []
        # storing all countors and exlude area less then 4 pixels
        for i in range(0, len(cts)):
            x, y, w, h = cv2.boundingRect(cts[i])
            # expanding the detected region to 120% for sub-window search
            x -= 0.3 * w
            y -= 0.5 * h
            w += 0.6 * w
            h += 1.0 * h
            if w > 2 and h > 2:
                contours.append(np.clip([round(x), round(y), round(x + w), round(y + h)], 0, map.shape[0]))
        # selecting all bboxes inside the contours
        # instead of picking bboxes out, we store the desired bboxes index
        target = np.zeros(500, )
        feat_bboxes = []
        count = 0
        num_boxes = 0
        # selected boxes index
        target = np.zeros(bboxes.shape[0], )
        for box in bboxes:
            # resized_box = box/(256/att_map.shape[0])
            x_min, y_min, x_max, y_max = box.cpu().data.numpy()
            for contour in contours:
                contour2 = contour * (256 / map.shape[0])
                # check if box is inside the contour
                if x_min >= contour2[0] and y_min >= contour2[1] and x_max <= contour2[2] and y_max <= contour2[3]:
                    target[count] = 1
                    num_boxes += 1
                    break
            count += 1
        return target, num_boxes

    def select_boxes(self, prob, rpn_box, targets):
        i = 0
        boxes = []
        box_probs = []
        for box in rpn_box:
            if targets[i] == 1.:
                boxes.append(box)
                box_probs.append(prob[i].data.cpu().numpy()[0])
            i += 1
        result_box = []
        indexs = np.argsort(box_probs)[-11: -1]
        for index in indexs:
            result_box.append(boxes[index])
        return result_box

    def forward(self, img, entity_one_hot, attr_one_hot):
        targets = np.zeros([self.opts.batch_size, self.visual_net.config.POST_NMS_ROIS_TRAINING])
        rpn_rois, concatenated_feat = self.visual_net(img)
        P3, P4, P5 = concatenated_feat[1], concatenated_feat[2], concatenated_feat[3]

        # Entity Branch
        entity_map, entity_feature = self.body(P3, P4, P5, entity_one_hot)

        # Attribute Branch
        attr_map, attr_feature = self.attr_branch(entity_feature, attr_one_hot)

        return entity_map, attr_map, attr_feature


class FrontBone(nn.Module):

    def __init__(self, entity_num):
        super(FrontBone, self).__init__()

        self.textual_emb = nn.Linear(entity_num, 256)
        self.mcb_p3 = CompactBilinearPooling(256, 256, 256).cuda()
        self.mcb_conv1_p3 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.mcb_relu1_p3 = nn.ReLU(inplace=True)
        self.mcb_conv2_p3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.mcb_p4 = CompactBilinearPooling(256, 256, 256).cuda()
        self.mcb_conv1_p4 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.mcb_relu1_p4 = nn.ReLU(inplace=True)
        self.mcb_conv2_p4 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.mcb_p5 = CompactBilinearPooling(256, 256, 256).cuda()
        self.mcb_conv1_p5 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.mcb_relu1_p5 = nn.ReLU(inplace=True)
        self.mcb_conv2_p5 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.mcb_sigmoid = nn.Sigmoid()

    def forward(self, P3, P4, P5, entity_one_hot):

        # Reshape entity one hot input
        entity_one_hot = self.textual_emb(entity_one_hot)
        entity_one_hot = entity_one_hot.view(entity_one_hot.shape[0], entity_one_hot.shape[1], 1, 1)

        # stack attention map generating for P3, P4, P5
        entity_one_hot = entity_one_hot.expand_as(P5)

        # Entity attention generation and applied
        mcb_feat5 = self.mcb_p5(entity_one_hot, P5)
        entity_map = self.mcb_sigmoid(self.mcb_conv2_p5(self.mcb_relu1_p5(self.mcb_conv1_p5(mcb_feat5))))
        entity_feature = (torch.mul(entity_map, P5))

        # Return: Object attention map, entity_feature
        return entity_map, entity_feature


class AttributeBranch(nn.Module):

    def __init__(self, attr_num):
        super(AttributeBranch, self).__init__()

        self.textual_emb = nn.Linear(attr_num, 256)
        self.mcb_p3 = CompactBilinearPooling(256, 256, 256).cuda()
        self.mcb_attr = CompactBilinearPooling(256, 256, 256).cuda()
        self.mcb_conv1_attr = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.mcb_relu1_attr = nn.ReLU(inplace=True)
        self.mcb_conv2_attr = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.mcb_sigmoid = nn.Sigmoid()

    def forward(self, entity_feature, attr_one_hot):

        # Reshape attribute one hot input
        attr_one_hot = self.textual_emb(attr_one_hot)
        attr_one_hot = attr_one_hot.view(attr_one_hot.shape[0], attr_one_hot.shape[1], 1, 1)

        # stack attention map generating for P3, P4, P5
        attr_one_hot = attr_one_hot.expand_as(entity_feature)

        # Attribute attention generation and applied
        mcb_attr_feat = self.mcb_attr(attr_one_hot, entity_feature)
        attr_map = self.mcb_sigmoid(self.mcb_conv2_attr(self.mcb_relu1_attr(self.mcb_conv1_attr(mcb_attr_feat))))
        attr_feature = (torch.mul(attr_map, entity_feature))

        # Return: attribute attention map, attribute feature
        return attr_map, attr_feature
