'''Train Sun Attribute with PyTorch.'''
from __future__ import print_function
from resnet import resnet101, resnet50
from tensorboardX import SummaryWriter
from coco_dataset import CocoCaptions
from torchvision import transforms
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from net_util import *
from parser import *
import random

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Randomly pick a label from multi one-hot label
def random_pick(one_hot):
    # return a randomly selected label
    label = torch.zeros(one_hot.shape[0])
    one_hot_return = torch.zeros_like(one_hot)

    for i in range(one_hot.shape[0]):
        # all labels to save all the labels
        all_labels = []
        count = 0
        for j in range(one_hot.shape[1]):
            if one_hot[i][j] == 1.:
                all_labels.append(count)
            count += 1
        # randomly picking one label
        if len(all_labels) != 0:
            label[i] = random.choice(all_labels)
        else:
            label[i] = 2
        one_hot_return[i][int(label[i])] = 1
    return label, one_hot_return


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

    for batch_idx, (images, attr_one_hot) in enumerate(data_loader):

        images = Variable(images).cuda()

        # Randomly pick one attribute per iteration
        single_attribute_label, single_attribute_one_hot = random_pick(attr_one_hot)
        single_attribute_label = Variable(single_attribute_label).long()
        conv_feat, y = net(images)

        # Computing the precision
        _, predicted = torch.max(y.data, 1)
        opts.correct += predicted.eq(single_attribute_label.data).cpu().sum()
        opts.total += y.data.shape[0]

        # Convert to float()
        attr_one_hot = Variable(attr_one_hot).cuda().float()

        # Cross-entropy loss or multi label cross-entropy loss
        loss = opts.criterion[0](y, attr_one_hot)

        train_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_idx += 1

        print('BCE: %.8f' % (loss.data))
        print('(random) Precision: %.8f' % (opts.correct/opts.total))
        if batch_idx % 10 == 0:
            writer.add_scalar('BCE Loss', train_loss / (batch_idx + 1), opts.iter_n)
            writer.add_scalar('Precision', opts.correct/opts.total, opts.iter_n)
            opts.iter_n += 10
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
        save_file_path = os.path.join(opts.checkpoint_path, 'AENet_clsfier_person_{}.pth'.format(opts.epoch))
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
    print("Preparing COCO data set...")
    opts.correct = 0
    opts.total = 0
    size = (512, 512)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    img_path = '/media/drive1/Data/coco17/train2017/'
    json = '/media/drive1/Data/coco17/annotations/captions_train2017.json'
    coco = COCO(json)

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    data_set = CocoCaptions(img_path, json, transform)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=opts.batch_size, shuffle=True)

    if not os.path.exists(opts.result_path):
        os.mkdir(opts.result_path)

    opts.train_epoch_logger = Logger(os.path.join(opts.result_path, 'train.log'),
                                     ['epoch', 'time', 'loss'])
    opts.train_batch_logger = Logger(os.path.join(opts.result_path, 'train_batch.log'),
                                     ['epoch', 'batch', 'loss'  ])
    opts.test_epoch_logger = Logger(os.path.join(opts.result_path, 'test.log'),
                                    ['epoch', 'time', 'loss'])

    # Model
    print('==> Building model...')
    model = resnet50(True, path='./checkpoint/AENet_clsfier_gender.pth', classnum=4)

    # Evaluation mode for batch normalization freeze
    # model.eval()
    # for p in model.parameters():
    #     p.requires_grad = True

    # Load Back bone Module
    # if opts.resume:
    #     state_dict = torch.load(opts.resume)['state_dict']
    #     new_params = model.state_dict()
    #     new_params.update(state_dict)
    #     model.load_state_dict(new_params)

    start_epoch = 0
    print('==> model built.')
    opts.criterion = [torch.nn.BCELoss()]

    # Training
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(params, 'trainable parameters in the network.')
    set_parameters(opts)
    opts.iter_n = 0

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