'''Train Sun Attribute with PyTorch.'''
from __future__ import print_function
from FlickrDataset2 import FlickrDataset2
from tensorboardX import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
from Model import Model
from Model2 import Model2
# from Model3 import Model3
from net_util import *
from parser import *
import math

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def l2_regulariza_loss(map):
    # return torch.mean(map.view(map.shape[0], map.shape[-2], map.shape[-1]))
    mean = torch.mean(map.view(map.shape[0], map.shape[-2], map.shape[-1]))
    return mean


def train_net(net, opts):

    print('training at epoch {}'.format(opts.epoch+1))

    if opts.use_gpu:
        net.cuda()

    net.train(True)
    train_loss = 0
    visual_train_loss = 0
    total_time = 0
    batch_idx = 0
    total = 0
    correct5 = 0
    correct4 = 0
    correct3 = 0
    optimizer = opts.current_optimizer
    back_bone_optimizer = opts.backbone_optimizer
    end_time = time.time()
    train_back_bone = True
    fig = plt.figure()

    for batch_idx, (images, category, (one_hot, label), textual_emb, phrase, mask, line, filename, size, all_one_hot,
                    all_attribute, att_label, all_line) in enumerate(data_loader):

        model.visual_net.config.IMAGES_PER_GPU = images.size(0)
        images = Variable(images).cuda()
        # BCE label
        all_one_hot = Variable(all_one_hot).cuda().float()
        label = Variable(label).cuda().long()

        # One-hot input
        one_hot = Variable(one_hot).cuda().float()

        # batch_boxes, category, att_map = net(images, one_hot, label)
        # loss = opts.criterion[0](category, label) + 0.01*l2_regulariza_loss(att_map)

        category_p3, att_map3, category_p4, att_map4, category_p5, att_map5, rpn_rois, visual_cls = net(images, one_hot, label)
        if train_back_bone:
            loss = opts.criterion[1](visual_cls, all_one_hot)
            back_bone_optimizer.zero_grad()
            visual_train_loss += loss.data[0]
            loss.backward()
            back_bone_optimizer.step()
            train_back_bone = not train_back_bone
            batch_idx += 1
            print('Visual BCE Loss: %.8f' % (2*visual_train_loss/(batch_idx+1)))

        else:
            # For Model3
            # category, att_map, rpn_rois = net(images, one_hot, label)
            # loss = opts.criterion[0](category, label)
            loss = opts.criterion[0](category_p3, label) + opts.criterion[0](category_p4, label) \
               + opts.criterion[0](category_p5, label)
            optimizer.zero_grad()  # flush
            _, predicted5 = torch.max(category_p5.data, 1)
            total += label.size(0)
            correct5 += predicted5.eq(label.data).cpu().sum()

            _, predicted4 = torch.max(category_p4.data, 1)
            correct4 += predicted4.eq(label.data).cpu().sum()

            _, predicted3 = torch.max(category_p3.data, 1)
            correct3 += predicted3.eq(label.data).cpu().sum()

            if not math.isnan(loss.data[0]):
                train_loss += loss.data[0]

            # Display the generated att_map and instant loss
            if batch_idx % 5 == 0:
                plt.ion()
                plt.show()
                random = randint(0, opts.batch_size - 1)
                if batch_idx % 1 == 0:
                    plt.suptitle(phrase[random] + ', Predicted3:' +class_names[predicted3[random]] +
                                 ', Predicted4:' +class_names[predicted4[random]] +
                                 ', Predicted5:' +class_names[predicted5[random]])
                    plt.subplot(141)
                    plt.imshow(att_map3[random, 0].data.cpu().numpy())
                    plt.subplot(142)
                    plt.imshow(att_map4[random, 0].data.cpu().numpy())
                    plt.subplot(143)
                    plt.imshow(att_map5[random, 0].data.cpu().numpy())
                    plt.subplot(144)
                    plt.imshow(images[random].permute(1, 2, 0).float().data.cpu())
                    plt.pause(0.001)
                    writer.add_scalar('Cross Entropy Loss', train_loss / (batch_idx+1), opts.iter_n)
                    opts.iter_n += 1

            print('Overall Loss: %.8f, Acc5: %.8f, Acc4: %.8f, Acc3: %.8f;'
                  % (2*train_loss/(batch_idx+1), correct3/total, correct4/total, correct5/total))

            if not math.isnan(loss.data[0]):
                loss.backward()
                optimizer.step()

            total_time += (time.time() - end_time)
            end_time = time.time()

            opts.train_batch_logger.log({
                'epoch': (opts.epoch+1),
                'batch': batch_idx+1,
                'loss': train_loss / (batch_idx+1),
            })

            train_back_bone = not train_back_bone
            batch_idx += 1

        if batch_idx % 3000 == 0:
            opts.train_epoch_logger.log({
                'epoch': (opts.epoch + 1),
                'loss': train_loss / (batch_idx + 1),
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
            save_file_path = os.path.join(opts.checkpoint_path,
                                          'Model2_flickr_P3-P4-P5_att_back-bone_fine-tune_clsfflex_{}.pth'.format(batch_idx))
            torch.save(net_states, save_file_path)

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
        save_file_path = os.path.join(opts.checkpoint_path, 'Model2_flickr_P3-P4-P5_att_back-bone_fine-tune_clsfflex_{}.pth'.format(opts.epoch))
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
    print(" Preparing Flickr data set...")
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
    model = Model2(opts)
    # Load Back bone Module
    if opts.resume:
        state_dict = torch.load(opts.resume)['state_dict']
        new_params = model.state_dict()
        new_params.update(state_dict)
        model.load_state_dict(new_params)
    start_epoch = 0
    print('==> model built.')
    opts.criterion = [torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()]
    # opts.criterion = [torch.nn.BCEWithLogitsLoss()]

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
            params = list(model.body.parameters()) + list(model.fc_p3.parameters()) + list(
                model.fc_p4.parameters()) + list(model.fc_p5.parameters()) + list(model.fc.parameters())
            visual_params = filter(lambda p: p.requires_grad, model.visual_net.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
            opts.backbone_optimizer = opts.optimizer(visual_params, lr=opts.lr/2, momentum=0.9, weight_decay=opts.weight_decay)

        elif (epoch % opts.lr_adjust_epoch) == 0 and epoch is not 0:
            opts.lr /= 10
            params = list(model.body.parameters()) + list(model.fc_p3.parameters()) + list(
                model.fc_p4.parameters()) + list(model.fc_p5.parameters()) + list(model.fc.parameters())
            visual_params = filter(lambda p: p.requires_grad, model.visual_net.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
            opts.backbone_optimizer = opts.optimizer(visual_params, lr=opts.lr/2, momentum=0.9, weight_decay=opts.weight_decay)

        train_net(model, opts)
        # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()