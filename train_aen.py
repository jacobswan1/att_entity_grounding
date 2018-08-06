'''Train Sun Attribute with PyTorch.'''
from __future__ import print_function
from FlickrDataset2 import FlickrDataset2
from tensorboardX import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
from resnet import resnet101
from net_util import *
from parser import *


def top_k_emb(visual_emb, text_emb, k=5):
    # Given pixel-wise and textual embeddings, select top-k most similar pixel embeddings out for
    # multi-instance learning
    # Visual-embedings:   (batch, pixels #, emb)
    # Textual-embeddings: (batch, emb)
    # Returning embeding: (batch #, att #, top-K, emb)
    visual_emb = visual_emb.view((visual_emb.shape[0], visual_emb.shape[1], visual_emb.shape[2] * visual_emb.shape[3]))
    return_emb = Variable(torch.zeros(visual_emb.shape[0], text_emb.shape[1], k, visual_emb.shape[1])).cuda()
    return_emb.requires_grad = True
    # i: batch number
    for i in range(visual_emb.shape[0]):
        # a: att number
        for a in range(text_emb.shape[1]):
            t_emb = text_emb[i][a]
            sorting = np.zeros((visual_emb.shape[2]))
            for j in range(visual_emb.shape[2]):
                sorting[j] = torch.nn.functional.cosine_similarity(visual_emb[i, :, j].contiguous().view(1, -1),
                                                                   t_emb.view(1, -1))
            # Arg-sort the cosine similarities matrix and inverse the order
            sorting = np.argsort(sorting)[::-1][0:k]
            # index: number of top-K
            for index in range(k):
                return_emb[i, a, index] = visual_emb[i, :, int(sorting[index])]

    return return_emb


def margin_loss(visual_feat, text_feat, label, margin=0.1):
    # Margin loss compute the loss as follows:
    # label: 0 or 1; v, t: target features; margin: minimum range between negative features
    # Loss = y(v-t)^2 + (1-y)(margin - ||v-t||)
    # visual_feat: (batch * attr #, top_K, pixel_emb)
    # text_feat:   (batch * attr #, pixel_emb)tsl
    # label:       (batch * attr #)
    diff = visual_feat - text_feat
    dist = (torch.mul(diff, diff).sum(1) / visual_feat.shape[1]).sqrt()
    zero = Variable(torch.from_numpy(np.zeros(1))).cuda().float()
    loss = label*dist + (1 - label)*torch.max(zero, margin - dist)

    return loss.sum(0)/loss.shape[0]


def train_net(net, opts):

    print('training at epoch {}'.format(opts.epoch+1))

    if opts.use_gpu:
        net.cuda()

    net.train(True)
    train_loss = 0
    total_time = 0
    batch_idx = 0
    optimizer = opts.current_optimizer
    end_time = time.time()
    fig = plt.figure()

    for batch_idx, (images, category, (one_hot, label), textual_emb, phrase, mask, line, filename, size, all_one_hot,
                    att_emb, att_label, ent_att_lable) in enumerate(data_loader):

        # model.visual_net.config.IMAGES_PER_GPU = images.size(0)
        images = Variable(images).cuda()
        att_label = Variable(att_label).cuda().float()
        att_emb = Variable(att_emb).cuda().float()

        conv_feat = net(images)

        # Pixel-embedding selecting and penalizing
        penalize_emb = top_k_emb(conv_feat, att_emb)
        # Reshape the visual pixel embeddings to (-1, emb)
        visual_feat = penalize_emb.view(-1, penalize_emb.shape[-1])

        # Reshape and expanding label from (batch * attr) to (batch * attr * top_K)
        label = att_label.view(-1, 1).expand(-1, penalize_emb.shape[2]).contiguous().view(
            att_label.shape[0] * att_label.shape[1] * penalize_emb.shape[2])

        # Expanding the textual embeddings to the same size with selected pixel_embeddings
        tmp = att_emb.view(att_emb.shape[0], att_emb.shape[1], 1, att_emb.shape[2])
        text_feat = tmp.expand_as(penalize_emb)
        text_feat = text_feat.contiguous().view(-1, text_feat.shape[-1])

        # Computing margin loss value
        loss = margin_loss(visual_feat, text_feat, label)

        train_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_idx += 1
        if batch_idx % 10 == 0:
            writer.add_scalar('BCE Loss', train_loss / (batch_idx + 1), opts.epoch*1500 +batch_idx)
        print('Margin Loss: %.8f' % (train_loss / (batch_idx + 1)))

    train_loss /= (batch_idx + 1)

    opts.train_epoch_logger.log({
        'epoch': (opts.epoch+1),
        'loss': train_loss,
        'time': total_time,
    })

    opts.train_losses.append(train_loss)

    # Save checkpoint.
    net_states = {
        'state_dict': net.state_dict(),
        'epoch': opts.epoch + 1,
        'loss': opts.train_losses,
        'optimizer': opts.current_optimizer.state_dict()
    }

    if opts.epoch % opts.checkpoint_epoch == 0:
        save_file_path = os.path.join(opts.checkpoint_path, 'AENet_P3_P4_clsfier_{}.pth'.format(opts.epoch))
        torch.save(net_states, save_file_path)

    print('Batch Loss: %.8f, elapsed time: %3.f seconds.' % (train_loss, total_time))


if __name__ == '__main__':

    opts = parse_opts()
    writer = SummaryWriter()

    if opts.gpu_id >= 0:
        torch.cuda.set_device(opts.gpu_id)
        opts.multi_gpu = False

    torch.manual_seed(opts.seed)
    if opts.use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed(opts.seed)

    # Loading Data
    print("Preparing Flickr data set...")
    size = (256, 256)
    feat_size = (128, 128)
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    data_set = FlickrDataset2('/media/drive1/Data/flickr30k/flickr30k_images/', feat_size, transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=opts.batch_size,
                                              shuffle=False)

    if not os.path.exists(opts.result_path):
        os.mkdir(opts.result_path)

    opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                                     ['epoch', 'time', 'loss'])
    opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                     ['epoch', 'batch', 'loss'])
    opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'),
                                    ['epoch', 'time', 'loss'])

    # Model
    print('==> Building model...')
    model = resnet101(True)
    model.fc = torch.nn.Linear(2048, 75)
    # Load Back bone Module
    if opts.resume:
        state_dict = torch.load(opts.resume)['state_dict']
        new_params = model.state_dict()
        new_params.update(state_dict)
        model.load_state_dict(new_params)
    start_epoch = 0
    print('==> model built.')
    # model.eval()
    opts.criterion = [torch.nn.BCELoss()]

    # Training
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params, 'trainable parameters in the network.')
    set_parameters(opts)
    opts.iter_n = 0
    # 81 classes loading
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    for epoch in range(start_epoch, start_epoch+opts.n_epoch):
        opts.epoch = epoch
        if epoch is 0:
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        elif (epoch % opts.lr_adjust_epoch) == 0 and epoch is not 0:
            opts.lr /= 10
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        train_net(model, opts)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()