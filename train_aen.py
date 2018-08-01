'''Train Sun Attribute with PyTorch.'''
from __future__ import print_function
from FlickrDataset2 import FlickrDataset2
from tensorboardX import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
from Model3 import Model3
from net_util import *
from parser import *


def margin_loss(visual_feat, text_feat, label, margin=0.1):
    # Margin loss compute the loss as follows:
    # label: 0 or 1; v, t: target features; margin: minimum range between negative features
    # Loss = y(v-t)^2 + (1-y)(margin - ||v-t||)
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

    for batch_idx, (images, category, (one_hot, label), textual_emb, phrase, mask, line, filename, size, all_one_hot, att_emb, att_label, all_line) \
            in enumerate(data_loader):

        model.visual_net.config.IMAGES_PER_GPU = images.size(0)
        images = Variable(images).cuda()
        all_one_hot = Variable(all_one_hot).cuda().float()
        att_emb = Variable(att_emb.view(att_emb.shape[0] * att_emb.shape[1], att_emb.shape[2]).float()).cuda()
        att_label = Variable(att_label.view(att_label.shape[0]*att_label.shape[1]).float()).cuda()

        v_feat, t_feat, att_map, rpn_rois, P3, P5 = net(images, all_one_hot, att_emb)

        loss = margin_loss(v_feat, t_feat, att_label)
        # Number of correctly predicted

        train_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_idx += 1
        print('Visual Margin Loss: %.8f' % (train_loss/(batch_idx+1)))

        # Display the generated att_map and instant loss
        if batch_idx % 5 == 0:
            plt.ion()
            plt.show()
            random = randint(0, opts.batch_size - 1)
            if batch_idx % 1 == 0:
                print(line)
                plt.subplot(121)
                plt.imshow(att_map[random, 0].data.cpu().numpy())
                plt.subplot(122)
                plt.imshow(images[random].permute(1, 2, 0).float().data.cpu())
                plt.pause(0.001)
                # writer.add_scalar('Cross Entropy Loss', train_loss / (batch_idx+1), opts.iter_n)
                opts.iter_n += 1
            print('Overall Loss: %.8f' % (train_loss/(batch_idx+1)))

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
        save_file_path = os.path.join(opts.checkpoint_path, 'AENet_P5_{}.pth'.format(opts.epoch))
        torch.save(net_states, save_file_path)

    print('Batch Loss: %.8f, elapsed time: %3.f seconds.' % (train_loss, total_time))


if __name__ == '__main__':

    opts = parse_opts()
    # writer = SummaryWriter()

    if opts.gpu_id >= 0:
        torch.cuda.set_device(opts.gpu_id)
        opts.multi_gpu = False

    torch.manual_seed(opts.seed)
    if opts.use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed(opts.seed)

    # Loading Data
    print("Preparing Flickr data set...")
    size = (1024, 1024)
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
    model = Model3(opts)
    # Load Back bone Module
    if opts.resume:
        state_dict = torch.load(opts.resume)['state_dict']
        new_params = model.state_dict()
        new_params.update(state_dict)
        model.load_state_dict(new_params)
    start_epoch = 2
    print('==> model built.')
    opts.criterion = [torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()]

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
            params = model.ae_net.parameters()
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        elif (epoch % opts.lr_adjust_epoch) == 0 and epoch is not 0:
            opts.lr /= 10
            params = model.ae_net.parameters()
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        train_net(model, opts)

    # writer.export_scalars_to_json("./all_scalars.json")
    # writer.close()