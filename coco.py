import torch.utils.data as data
from PIL import Image
import os
import random
import os.path
from pycocotools.coco import COCO
import numpy as np


class CocoCaptions(data.Dataset):
    """`MS Coco Captions <http://mscoco.org/dataset/#captions-challenge2015>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    Example:
        .. code:: python
            import torchvision.datasets as dset
            import torchvision.transforms as transforms
            cap = dset.CocoCaptions(root = 'dir where images are',
                                    annFile = 'json annotation file',
                                    transform=transforms.ToTensor())
            print('Number of samples: ', len(cap))
            img, target = cap[3] # load 4th sample
            print("Image Size: ", img.size())
            print(target)
        Output: ::
            Number of samples: 82783
            Image Size: (3L, 427L, 640L)
            [u'A plane emitting smoke stream flying over a mountain.',
            u'A plane darts across a bright blue sky behind a mountain covered in snow',
            u'A plane leaves a contrail above the snowy mountain top.',
            u'A mountain that has a plane flying overheard in the distance.',
            u'A mountain view with a plume of smoke in the background']
    """

    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        print('loading dictionary...')
        embeddings_index = {}
        f = open('glove.6B.' + str(300) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            embeddings_index[word] = coefs
        f.close()
        print('dictionary loaded.')
        self.embeddings_index = embeddings_index

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # embedding words: 32, size: 300
        embedding = np.zeros((32, 300))
        count = 0
        for sent in target:
            array = sent.replace('.', '').split(' ')
            for i in range(len(array)):
                if count == 32:
                    break
                if len(array[i].lower()) is not 0:
                    try:
                        embedding[count] = self.embeddings_index[array[i].lower()]
                        count += 1
                    except:
                        break
        img = np.asarray(img)
        return img, target, embedding

    def __len__(self):
        return len(self.ids)


class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, coco, root, transform=None, target_transform=None):
        self.root = root
        self.coco = coco
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cats = coco.loadCats(coco.getCatIds())

    def __getitem__(self, img_id):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        img = np.asarray(img)

        # random picking one object
        size = len(target)
        if size != 0:
            index = random.randint(0, len(target) - 1)
            bbox = target[index]['bbox']
            cat_id = target[index]['category_id']
            one_hot, label, name = self.convert_one_hot(cat_id, self.cats)
        else:
            bbox = []
            cat_id = 0
            one_hot, label, name = self.convert_one_hot(cat_id, self.cats)

        # Return all labels
        all_one_hot = np.zeros(81, )
        for item in target:
            arr_id = 1
            cat_label = item['category_id']
            for item in self.cats:
                if cat_label == item['id']:
                    break
                arr_id += 1
            all_one_hot[arr_id] += 1
        return img, label, one_hot, name, all_one_hot

    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def convert_one_hot(self, cat_id, cats):
        one_hot = np.zeros(81, )
        name = ''
        if cat_id != 0:
            count = 1
            for item in cats:
                if cat_id == item['id']:
                    name = item['name']
                    break
                count += 1
            one_hot[count] = 1
        else:
            count = cat_id
            one_hot[cat_id] = 1
        return one_hot, count, name