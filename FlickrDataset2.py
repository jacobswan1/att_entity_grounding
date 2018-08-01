# Retrieve annotation from list
# new Dataset Class
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import pickle
import re
import nltk


class FlickrDataset2(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, feat_size, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # read image directories
        list_file = open('img_list.txt', 'r')
        img_list = []
        for i in list_file.readlines():
            img_list.append(i.replace('\n', ''))

        # read from annotations
        with open('annotations.pkl', 'rb') as f:
            labels = pickle.load(f)

        # Load Word Embedding
        print('Loading dictionary...')
        embeddings_index = {}
        f = open('glove.6B.' + str(300) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()
        print('Dictionary loaded.')

        self.embeddings_index = embeddings_index
        self.labels = labels
        self.files = img_list
        self.length = len(self.files)
        self.root_dir = root_dir
        self.transform = transform
        self.feat_size = feat_size

    def __len__(self):
        return self.length

    def pos_neg_pairs(self, attr, frequent_atts, length=8):
        # Construct a neg-pos attributes set with the length to be default as 8
        # returning example:  (['women', 'women', 'women', 'women', 'wet'], array([1., 1., 1., 1., 0.]))
        # 1 represents existing positibe attribute while 0 refers to randomly picked non-existing att
        # we replace the word to 300d word-embedding in this case
        attributes = np.zeros((length, 300))
        label = np.zeros(length, )
        if len(attr) >= length:
            for i in range(length):
                attributes[i] = self.embeddings_index[attr[i]]
            label[0: length] = 1
        else:
            l = len(attr)
            for i in range(l):
                attributes[i] = self.embeddings_index[attr[i]]
                label[0:l] = 1
            # For remaining attribute we randomly select attributes from ictionary
            for i in range(length - l):
                state = True
                while state:
                    ran = random.choice(frequent_atts)
                    if ran not in attributes:
                        attributes[l + i] = self.embeddings_index[ran]
                        state = False
        return attributes, label

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
        # Label_pair consists of one-hot&label_num
        label_pair = label[1]
        textual_emb = label[2]
        text = label[3]
        size = label[4][0]
        boxes = label[4][1]
        line = label[5]
        mask, overal_box = generate_mask(np.asarray(size), self.feat_size, np.asarray(boxes))

        # Labels for all objects, and captions
        all_one_hot = np.zeros(81, )
        all_line = ''
        for item in labels:
            all_one_hot += labels[item][1][0]
            all_line += labels[item][-1]

        # Extract all existing attributes
        attr = []
        regex = r'\[.*?\]'
        text_regex = r'\s.+\]'
        append_atts = ['men', 'man', 'male', 'female', 'women', 'woman', 'girl', 'boy', 'older']
        delete_atts = ['other', '"', 'third', 'second', 'first', 'sure', 'ping-pong', 'palid']
        # Read frequent attrivutes in flickr 30k
        att_dict = open('att_dict_flickr.txt', 'r')
        frequent_atts = []
        for i in att_dict.readlines():
            frequent_atts.append(i.replace('\n', ''))

        value = re.findall(regex, all_line)
        for item in value:
            phrase = re.findall(text_regex, item)[0][0:-1]
            words = nltk.pos_tag([item for item in phrase.replace('-', ' ').split(' ') if len(item) > 0])
            for word in words:
                if (word[1] == 'JJ' or word[1] == 'NN' or word[0] in append_atts) and word[0] not in delete_atts:
                    if word[0] not in attr and word[0] in self.embeddings_index.keys():
                        attr.append(word[0].lower())

        attr_emb, attr_label = self.pos_neg_pairs(attr, frequent_atts)

        # Extract all attribute & entity for classification
        # Load att-entity dict
        list_file = open('entity_att_flickr.txt', 'r')
        entity_att = []
        for i in list_file.readlines():
            entity_att.append(i.replace('\n', ''))
        ent_att_lable = np.zeros(75)
        phrase = re.findall(regex, all_line)
        for p in phrase:
            words = re.findall(text_regex, p)[0][0:-1]
            words = nltk.pos_tag([item for item in words.split(' ') if len(item) > 0])
            for item in words:
                word = item[0]
                att = item[1]
                # NNS case
                if att == 'NNS':
                    word = word[0:-1]
                    if word.lower() in entity_att:
                        index = entity_att.index(word.lower())
                        ent_att_lable[index] = 1
                # attribute or entity case
                elif att == 'JJ' or att == 'NN':
                    if word.lower() in entity_att:
                        index = entity_att.index(word.lower())
                        ent_att_lable[index] = 1

        return img, category, label_pair, textual_emb, text, mask, line, filename, size, all_one_hot, attr_emb, attr_label, ent_att_lable


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

