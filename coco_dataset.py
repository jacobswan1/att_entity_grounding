import os
import torch
import nltk
import random
import os.path
import numpy as np
from PIL import Image
import torch.utils.data as data
from pycocotools.coco import COCO
from torchvision import transforms


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

        # Load COCO image IDs
        list_file = open('coco_person_list.txt', 'r')
        ids = []
        for i in list_file.readlines():
            ids.append(int(i.replace('\n', '')))

        # Load entity-attribute dictionary
        entity_att = []
        list_file = open('low-level-attr.txt', 'r')
        for i in list_file.readlines():
            entity_att.append(i.replace('\n', ''))

        self.ids = ids
        self.coco = COCO(annFile)
        self.entity_att_dict = entity_att
        self.transform = transform
        self.root = os.path.expanduser(root)
        self.target_transform = target_transform

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

        img = np.asarray(img)
        ent_att_lable = np.zeros(4)

        for sentence in target:
            words = nltk.pos_tag([item for item in sentence.replace('.', ' ').split(' ') if len(item) > 0])
            for item in words:
                word = item[0].lower()
                # att = item[1]
                if word in self.entity_att_dict:
                    att_id = self.entity_att_dict.index(word)
                    ent_att_lable[att_id] = 1

        return img, ent_att_lable

    def __len__(self):
        return len(self.ids)


# if __name__ == '__main__':
#     size = (512, 512)
#     img_path = '/media/drive1/Data/coco17/train2017/'
#     json = '/media/drive1/Data/coco17/annotations/captions_train2017.json'
#     coco = COCO(json)
#     transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
#     data_set = CocoCaptions(img_path, json, transform)
#     data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False)
#
#     img_ids = []
#     count = 1
#     for index, (img, target) in enumerate(data_loader):
#         print(target)
