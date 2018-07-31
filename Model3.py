# Grounding Model with attribute embedding

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


class Model3(nn.Module):

    def __init__(self, opts, body_pretrain=False):
        super(Model3, self).__init__()
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
        # Freeze the back_bone model
        for param in visual_net.parameters():
            param.requires_grad = False

        # Freeze the classifier
        for param in visual_net.fpn_classifier_graph.parameters():
            param.requires_grad = False

        # Load Front bone module
        body = FrontBone(opts.class_num)
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

        self.body = body
        self.opts = opts
        self.visual_net = visual_net
        self.visual_net.eval()
        self.body.eval()
        self.ae_net = AENet2()

        # feature expanding to 1024 then feed to mask-RCNN classifier
        self.fc_p3 = nn.Linear(256, 1024)
        self.fc_p4 = nn.Linear(256, 1024)
        self.fc_p5 = nn.Linear(256, 1024)
        self.fc = nn.Linear(256, 81)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.pool = nn.AvgPool2d(64)

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

    def forward(self, img, one_hot, embedding):
        targets = np.zeros([self.opts.batch_size, self.visual_net.config.POST_NMS_ROIS_TRAINING])
        rpn_rois, concatenated_feat = self.visual_net(img)
        P3, P4, P5 = concatenated_feat[1], concatenated_feat[2], concatenated_feat[3]
        # Stack-attention and features
        att_map3, att_feat3, att_map4, att_feat4, att_map5, att_feat5 = self.body(P3, P4, P5, one_hot)

        # Concatenate Attention Maps
        # con_map = (att_map3 + F.upsample(att_map5, scale_factor=2, mode='bilinear'))/2
        con_map = (nn.functional.avg_pool2d(att_map3, 2) + att_map5)/2
        v_feat, t_feat = self.ae_net(P3, P5, con_map, embedding)

        return v_feat, t_feat, con_map, rpn_rois, P5


class FrontBone(nn.Module):

    def __init__(self, class_num):
        super(FrontBone, self).__init__()

        self.textual_emb = nn.Linear(class_num, 256)
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
        self.instance_norm = torch.nn.InstanceNorm2d(256)
        self.mcb_pool_p3 = nn.AvgPool2d(128)
        self.mcb_pool_p4 = nn.AvgPool2d(64)
        self.mcb_pool_p5 = nn.AvgPool2d(64)

    def sign_norm(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def forward(self, P3, P4, P5, one_hot):

        one_hot = self.textual_emb(one_hot)
        one_hot = one_hot.view(one_hot.shape[0], one_hot.shape[1], 1, 1)

        # stack attention map generating for P3, P4, P5
        one_hot3 = one_hot.expand_as(P3)
        one_hot4 = one_hot.expand_as(P4)
        one_hot5 = one_hot.expand_as(P5)

        # P3
        mcb_feat3 = self.mcb_p3(one_hot3, P3)
        att_map3 = self.mcb_sigmoid(self.mcb_conv2_p3(self.mcb_relu1_p3(self.mcb_conv1_p3(mcb_feat3))))
        att_feat3 = self.mcb_pool_p3(torch.mul(att_map3, P3))
        att_feat3 = att_feat3.view(att_feat3.shape[0], -1)

        # P4
        mcb_feat4 = self.mcb_p4(one_hot4, P4)
        att_map4 = self.mcb_sigmoid(self.mcb_conv2_p4(self.mcb_relu1_p4(self.mcb_conv1_p4(mcb_feat4))))
        att_feat4 = self.mcb_pool_p4(torch.mul(att_map4, P4))
        att_feat4 = att_feat4.view(att_feat4.shape[0], -1)

        # P5
        mcb_feat5 = self.mcb_p5(one_hot5, P5)
        att_map5 = self.mcb_sigmoid(self.mcb_conv2_p5(self.mcb_relu1_p5(self.mcb_conv1_p5(mcb_feat5))))
        att_feat5 = self.mcb_pool_p5(torch.mul(att_map5, P5))
        att_feat5 = att_feat5.view(att_feat5.shape[0], -1)

        # return: res_net_class, model_class, attention_map, attention_feature
        return att_map3, att_feat3, att_map4, att_feat4, att_map5, att_feat5


class AENet(nn.Module):
    def __init__(self):
        super(AENet, self).__init__()
        self.avgpool = nn.AvgPool2d(64)
        self.visual_fc = nn.Linear(256, 1024)
        self.text_fc = nn.Linear(300, 1024)
        self.conv = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=1, bias=True)
        self.prelu = nn.PReLU()

    def forward(self, P5, entity_map, embedding):
        text_feat = self.prelu(self.text_fc(embedding))
        visual_feat = self.avgpool(torch.mul(P5, entity_map))
        visual_feat = visual_feat.view(visual_feat.shape[0], visual_feat.shape[1])
        visual_feat = self.prelu(self.visual_fc(visual_feat))

        # Expand from (batch, 1024) to (batch* for pos-neg margin loss
        visual_feat = visual_feat.view(visual_feat.shape[0], 1, visual_feat.shape[1]).expand(visual_feat.shape[0], 8, visual_feat.shape[1])\
            .contiguous().view(visual_feat.shape[0] * 8, visual_feat.shape[1])
        return visual_feat, text_feat


class AENet2(nn.Module):
    def __init__(self):
        super(AENet2, self).__init__()
        self.avgpool = nn.AvgPool2d(64)
        self.visual_fc = nn.Linear(512, 1024)
        self.text_fc = nn.Linear(300, 1024)
        self.conv = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True)
        self.prelu = nn.PReLU()
        self.relu = nn.ReLU()

    def forward(self, P3, P5, entity_map, embedding):
        text_feat = self.prelu(self.text_fc(embedding))

        # P3 & P5 concatenation
        P5 = torch.cat([torch.nn.functional.avg_pool2d(P3, 2), P5], 1)
        P5 = self.relu(self.conv(P5))
        visual_feat = self.avgpool(torch.mul(P5, entity_map))
        visual_feat = visual_feat.view(visual_feat.shape[0], visual_feat.shape[1])
        visual_feat = self.prelu(self.visual_fc(visual_feat))

        # Expand from (batch, 1024) to (batch* for pos-neg margin loss
        visual_feat = visual_feat.view(visual_feat.shape[0], 1, visual_feat.shape[1]).expand(visual_feat.shape[0], 8, visual_feat.shape[1])\
            .contiguous().view(visual_feat.shape[0] * 8, visual_feat.shape[1])
        return visual_feat, text_feat