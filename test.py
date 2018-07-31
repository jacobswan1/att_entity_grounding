'''Flickr 30k Grounding with PyTorch.'''
from FlickrDataset2 import FlickrDataset2
from tensorboardX import SummaryWriter
import matplotlib.patches as patches
from torch.autograd import Variable
from torchvision import transforms
from skimage.morphology import *
import matplotlib.pyplot as plt
from Model3 import Model3
from net_util import *
from parser import *
import cv2
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def retrieve_bboxes(att_map, bboxes):
    # compute the mean value of attention map
    mean = np.mean(att_map)
    # transform to 0-255 scale image
    test = (att_map * 255).astype('uint8')
    # threshold set to 3 times of mean value
    ret,thresh = cv2.threshold(test,round(mean*3*255), 255, 0)
    # contour detection
    im2, cts, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    # storing all countors and exlude area less then 4 pixels
    for i in range(0,len(cts)):
        x, y, w, h = cv2.boundingRect(cts[i])
        # expanding the detected region to 120% for sub-window search
        x -= 0.4*w
        y -= 0.4*h
        w += .8*w
        h += .8*h
        if w>2 and h>2:
            contours.append(np.clip([round(x), round(y), round(x+w), round(y+h)], 0, att_map.shape[0]))
    # selecting all bboxes inside the contours
    # instead of picking bboxes out, we store the desired bboxes index
    target = np.zeros(500,)
    feat_bboxes = []
    count = 0
    num_boxes = 0
    # selected boxes index
    print(contours)
    target = np.zeros(bboxes.shape[0],)
    for box in bboxes:
        x_min, y_min, x_max, y_max = box.cpu().data.numpy()
        for contour in contours:
            contour2 = contour * (1024/att_map.shape[0])
            # check if box is inside the contour
#             print(x_min, y_min, w, h)
            if x_min >= contour2[0] and y_min >= contour2[1] and x_max <= contour2[2] and y_max <= contour2[3]:
                target[count] = 1
                num_boxes += 1
                break
        count += 1
    return target, num_boxes


# Load Pretrained Model.
class opts():
    backbone_model = './models/mrcnn.pth'
    class_num = 81
    batch_size = 64
    resume = './checkpoint/AENet_0.pth'


opts = opts()
model = Model3(opts)
# Load Back bone Module
state_dict = torch.load(opts.resume)['state_dict']
new_params = model.state_dict()
new_params.update(state_dict)
model.load_state_dict(new_params)
model.cuda()
model.eval()
print('Model loaded')
# Load dataset and images
print("Preparing Flickr data set...")
size = (1024, 1024)
feat_size = (128, 128)
transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
data_set = FlickrDataset2('/media/drive1/Data/flickr30k/flickr30k_images/', feat_size, transform)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=1,shuffle=False)
for batch_idx, (images, category, (one_hot, label), textual_emb, phrase, mask, line, filename, size, all_one_hot, att_emb, att_label) \
        in enumerate(data_loader):

    model.visual_net.config.IMAGES_PER_GPU = images.size(0)
    images = Variable(images).cuda()
    all_one_hot = Variable(all_one_hot).cuda().float()
    one_hot = Variable(one_hot).cuda().float()
    att_emb = Variable(att_emb.view(att_emb.shape[0] * att_emb.shape[1], att_emb.shape[2]).float()).cuda()
    att_label = Variable(att_label.view(att_label.shape[0]*att_label.shape[1]).float()).cuda()
    model.visual_net.config.IMAGES_PER_GPU = 1
    v_feat, t_feat, att_map = model(images, one_hot, att_emb)
