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


class Model(nn.Module):

    def __init__(self, opts, body_pretrain=False):
        super(Model, self).__init__()
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
        visual_net.eval()
        print(visual_net.config)
        for param in visual_net.parameters():
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

        # feature expanding to 1024 then feed to mask-RCNN classifier
        self.fc = nn.Linear(256, 1024)

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

    def forward(self, img, one_hot, labels):
        targets = np.zeros([self.opts.batch_size, self.visual_net.config.POST_NMS_ROIS_TRAINING])
        rpn_rois, concatenated_feat = self.visual_net(img)
        conv_feat = concatenated_feat[-1]
        att_map, att_feat = self.body(conv_feat, one_hot)
        att_feat = F.relu(self.fc(att_feat))
        category = self.visual_net.fpn_classifier_graph(att_feat, rpn_rois, False)
        # pass concatenated visual feat to classifier for later picking
        mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = self.visual_net.fpn_classifier_graph(concatenated_feat, rpn_rois, True)

        # retrieve bounding boxes
        # att = 1 - att_map[:, 0].data.cpu().numpy()
        att = att_map[:, 0].data.cpu().numpy()
        for i in range(img.size(0)):
            targets[i], num_boxes = self.retrieve_bboxes(att[i], rpn_rois[i])

        # select boxes that contained in attention map
        count = 0
        batch_boxes = []
        for item in range(mrcnn_probs.shape[0]):
            probs = mrcnn_probs[item]
            # retrieve class possibility for the current slice
            prob = probs[:, labels[item]]
            boxes = self.select_boxes(prob, rpn_rois[count].data, targets[count])
            batch_boxes.append(boxes)
            count += 1

        # return: res_net_class, model_class, attention_map, attention_feature

        return batch_boxes, category, att_map, rpn_rois


class FrontBone(nn.Module):

    def __init__(self, class_num):
        super(FrontBone, self).__init__()

        self.textual_emb = nn.Linear(class_num, 256)
        self.mcb = CompactBilinearPooling(256, 256, 256).cuda()
        self.mcb_conv1 = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.mcb_relu1 = nn.ReLU(inplace=True)
        self.mcb_conv2 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.mcb_sogmoid = nn.Sigmoid()
        self.mcb_pool = nn.AvgPool2d(64)
        # self.fc = nn.Linear(256, class_num)

    def forward(self, conv_feat, one_hot):

        one_hot = self.textual_emb(one_hot)
        one_hot = one_hot.view(one_hot.shape[0], one_hot.shape[1], 1, 1)

        one_hot = one_hot.expand_as(conv_feat)
        mcb_feat = self.mcb(one_hot, conv_feat)

        att_map = self.mcb_sogmoid(self.mcb_conv2(self.mcb_relu1(self.mcb_conv1(mcb_feat))))
        att_feat = self.mcb_pool(torch.mul(1-att_map, conv_feat))
        att_feat = att_feat.view(att_feat.shape[0], -1)
        # category = self.fc(att_feat)

        # return: res_net_class, model_class, attention_map, attention_feature
        return att_map, att_feat