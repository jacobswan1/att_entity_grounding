# Dataset Class
from torch.utils.data import Dataset
from torchvision import transforms
import xml.etree.ElementTree
from PIL import Image
import numpy as np
import random
import re
import os


class FlickrDateset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, feat_size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        xml_root = '../Data/flickr30k/Flickr30kEntities/Annotations/'
        text_root = '../Data/flickr30k/Flickr30kEntities/Sentences/'
        list_file = open('img_list.txt', 'r')
        img_list = []
        for i in list_file.readlines():
            img_list.append(i.replace('\n', ''))
        files = os.listdir(xml_root)
        num_regex = r'\d+'
        category_regex = r'\/\w+\s'
        text_regex = r'\s.+\]'
        labels = {}

        for file in files:
            # E_freq for storing the words and random picking
            entity_freq = []
            f = file.replace('.xml', '')
            f2 = f + '.jpg'
            xml_path = xml_root + f + '.xml'
            text_path = text_root + f + '.txt'

            # Parse xml file to retrieve bounding boxes
            e = xml.etree.ElementTree.parse(xml_path)
            objects = {}
            annotations = {}
            for m in e.findall('object'):
                coords = []
                size = []
                name = int(m.find('name').text)
                entity_freq.append(name)
                if e.findall('size') is not None:
                    size.append(int(e.find('size').find('width').text))
                    size.append(int(e.find('size').find('height').text))
                if m.find('bndbox') is not None:
                    coords.append(int(m.find('bndbox').find('xmin').text))
                    coords.append(int(m.find('bndbox').find('ymin').text))
                    coords.append(int(m.find('bndbox').find('xmax').text))
                    coords.append(int(m.find('bndbox').find('ymax').text))
                    if name not in objects.keys():
                        objects[name] = (size, [coords])
                    else:
                        objects[name][1].append(coords)

            # Parse txt file to retrieve texts and embeddings
            lines = open(text_path, 'r').read().split('\n')

            for line in lines:
                # Parse per line
                text_set = []
                regex = r'\[.*?\]'
                value = re.findall(regex, line)
                text_set.append(value)
                for t_set in text_set[0]:
                    text_id = int(re.findall(num_regex, t_set)[0])
                    category = re.findall(category_regex, t_set)[0][1:-1]

                    # Get text embeddings
                    phrase = re.findall(text_regex, t_set)[0][1:-1]
                    phrase = phrase.replace('-', ' ')

                    if text_id in objects.keys() and category != 'scene' and category != 'bodyparts':
                        annotations[text_id] = (category, phrase, objects[text_id], line)

            # Exclude images that only contains "other" categor
            if len(annotations.keys()) != 0:
                labels[f] = [annotations, entity_freq, file]

        self.labels = labels
        # self.files = os.listdir(root_dir)
        self.files = img_list
        self.length = len(self.files)
        self.root_dir = root_dir
        self.transform = transform
        self.feat_size = feat_size

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        image = Image.open(self.root_dir + self.files[index])
        image = image.convert('RGB')
        if self.transform is not None:
            img = self.transform(image)

        labels = self.labels[self.files[index].replace('.jpg', '')][0]
        entity_freq = self.labels[self.files[index].replace('.jpg', '')][1]
        filename = self.labels[self.files[index].replace('.jpg', '')][2]
        try:
            label = labels[random.choice(entity_freq)]
        except:
            index = random.choice(list(labels.keys()))
            label = labels[index]

        category = label[0]
        text = label[1]
        size = label[2][0]
        boxes = label[2][1]
        line = label[3]
        mask, overal_box = generate_mask(np.asarray(size), self.feat_size, np.asarray(boxes))

        return img, text, mask, category, overal_box, one_hot_category(category), line, filename


def generate_mask(ori_size, size, coords):
    ratio = size / ori_size
    ratio = np.array([ratio[0], ratio[1], ratio[0], ratio[1]])
    coords = coords * ratio
    overal_box = get_overalbox(coords)
    mask = np.zeros(size)
    for box in coords:
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if (box[0] < x < box[2]) and (box[1] < y < box[3]):
                    mask[y, x] = 1
    return mask, overal_box


def get_overalbox(coods):
    x_min = coods[:, 0].min()
    y_min = coods[:, 1].min()
    x_max = coods[:, 2].max()
    y_max = coods[:, 3].max()
    overal_box = [x_min, y_min, x_max, y_max]
    return overal_box


def one_hot_category(label):
    one_hot = np.zeros(4)
    # cls_dic = {'animals': 0,
    #            'bodyparts': 1,
    #            'clothing': 2,
    #            'instruments': 3,
    #            'people': 4,
    #            'scene': 5,
    #            'vehicles': 6}
    cls_dic = {'animals': 0,
               'clothing': 1,
               'people': 2,
               'vehicles': 3}
    label_num = cls_dic[label]
    one_hot[label_num] = 1
    return one_hot, label_num


if __name__ == '__main__':
    size = (256, 256)
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    dataset = FlickrDateset('/media/drive1/Data/flickr30k/flickr30k_images/', (32, 32), transform)
    for i in range(1000):
        dataset[i]