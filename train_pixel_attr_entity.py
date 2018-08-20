'''Train unsuperwised entity grounding by attention+pixel classification mechanism.'''
from __future__ import print_function
from FlickrDataset2 import FlickrDataset2
from tensorboardX import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Model6 import Model6
from net_util import *
from parser import *
import math

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Multi-Pixel embedding learning for multi-category picking
def top_k_emb(visual_emb, model, label, K=100):
    # Given pixel-wise features, select top-k pixels with highest category prob out for
    # multi-cross entropy learning
    # Visual-features:   (batch, emb #, pixel #)
    # Returning prob: (batch #, top-K, class_prob)
    visual_emb = visual_emb.view((visual_emb.shape[0], visual_emb.shape[1], visual_emb.shape[2]*visual_emb.shape[3]))
    prob_set = Variable(torch.zeros(visual_emb.shape[2], 75)).cuda()
    return_prob = Variable(torch.zeros(visual_emb.shape[0], K, 75)).cuda()
    return_prob.requires_grad = True
    prob_set.requires_grad = True
    # i: batch number
    for i in range(visual_emb.shape[0]):
        sorting = np.zeros((visual_emb.shape[2]))

        # j: pixel numbers in feature maps
        for j in range(visual_emb.shape[2]):
            # extracting pixel features and reshape
            emb = visual_emb[i,:,j]
            emb = F.relu(model.fc_p5(emb.contiguous().view(1, -1)))
            output = torch.nn.functional.sigmoid(model.fc(emb))
            prob = torch.nn.functional.binary_cross_entropy(output.view(-1), label[i])
            prob_set[j] = output[0]
            l = label[i]
            sorting[j] = prob

        # Arg-sort the cosine similarities matrix (and inverse the order)
        # sorting = np.argsort(sorting)[::-1][0:K]
        sorting = np.argsort(sorting)[0:K]
        # index: number of top-K
        for index in range(K):
            return_prob[i, index] = prob_set[int(sorting[index])]

    return return_prob


def train_net(net, opts):

    print('training at epoch {}'.format(opts.epoch+1))

    if opts.use_gpu:
        net.cuda()

    net.train(True)
    k = 100
    train_loss = 0
    total_time = 0
    batch_idx = 0
    optimizer = opts.current_optimizer
    # back_bone_optimizer = opts.backbone_optimizer
    end_time = time.time()
    train_back_bone = True
    fig = plt.figure()

    for batch_idx, (images, category, (one_hot, label), textual_emb, phrase, mask, line, filename, size, all_one_hot,
                    att_emb, att_label, attr_one_hot) in enumerate(data_loader):

        model.visual_net.config.IMAGES_PER_GPU = images.size(0)
        images = Variable(images).cuda()

        # One-hot input
        attr_one_hot = Variable(attr_one_hot).cuda().float()
        entity_one_hot = Variable(all_one_hot).cuda().float()

        # Feed in network
        entity_map, attr_map, attr_feature = net(images, entity_one_hot, attr_one_hot)

        # Pixel Multi-Instance Learning
        pixel_prob = top_k_emb(attr_feature, model, attr_one_hot, k)
        pixel_prob = pixel_prob.view(-1, pixel_prob.shape[-1])

        # Expanding the ground truth label for classification
        label = attr_one_hot.view(attr_one_hot.shape[0], 1, -1).expand(attr_one_hot.shape[0], k,
                                                                       attr_one_hot.shape[-1]).contiguous().view(-1, attr_one_hot.shape[-1])
        loss = opts.criterion[0](pixel_prob, label)

        optimizer.zero_grad()  # flush
        # _, predicted5 = torch.max(pixel_prob.data, 1)
        # total += label.size(0)
        # correct5 += predicted5.eq(label.data).cpu().sum()

        if not math.isnan(loss.data[0]):
            train_loss += loss.data[0]

        # Display the generated att_map and instant loss
        if batch_idx % 2 == 0:
            plt.ion()
            plt.show()
            random = randint(0, opts.batch_size - 1)
            if batch_idx % 1 == 0:
                # Print out the attribute labels
                t = 0
                printing = ''
                for i in attr_one_hot[random]:
                    if i == 1.0:
                        printing += entity_att[t] + ' '
                    t += 1
                plt.suptitle(printing)

                plt.subplot(131)
                plt.imshow(entity_map[random, 0].data.cpu().numpy())
                plt.subplot(132)
                plt.imshow(entity_map[random, 0].data.cpu().numpy()*attr_map[random, 0].data.cpu().numpy())
                plt.subplot(133)
                plt.imshow(images[random].permute(1, 2, 0).float().data.cpu())
                plt.pause(0.001)
                writer.add_scalar('Cross Entropy Loss', train_loss / (batch_idx+1), opts.iter_n)
                opts.iter_n += 1

        print('Overall Loss: %.8f'
              % (train_loss/(batch_idx+1)))

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

        if batch_idx % 1000 == 0:
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
                                          'Model5_flickr_P5_attr_{}.pth'.format(batch_idx))
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
        save_file_path = os.path.join(opts.checkpoint_path, 'Model6_flickr_P5_attr_entity_pixel_cls_{}.pth'.format(opts.epoch))
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
    size = (1024, 1024)
    feat_size = (128, 128)
    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])
    data_set = FlickrDataset2('/media/drive1/Data/flickr30k/flickr30k_images/', feat_size, transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=opts.batch_size,
                                              shuffle=False)

    # Loading label annotations
    list_file = open('entity_att_flickr.txt', 'r')
    entity_att = []
    for i in list_file.readlines():
        entity_att.append(i.replace('\n', ''))

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
    model = Model6(opts)

    # Load Back bone Module
    if opts.resume:
        state_dict = torch.load(opts.resume)['state_dict']
        new_params = model.state_dict()
        new_params.update(state_dict)
        model.load_state_dict(new_params)
    start_epoch = 0
    print('==> model built.')
    opts.criterion = [torch.nn.BCEWithLogitsLoss()]

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
            # opts.backbone_optimizer = opts.optimizer(visual_params, lr=opts.lr/2, momentum=0.9,
            # weight_decay=opts.weight_decay)

        elif (epoch % opts.lr_adjust_epoch) == 0 and epoch is not 0:
            opts.lr /= 10
            params = filter(lambda p: p.requires_grad, model.parameters())
            opts.current_optimizer = opts.optimizer(params, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
            # opts.backbone_optimizer = opts.optimizer(visual_params, lr=opts.lr/2, momentum=0.9,
            # weight_decay=opts.weight_decay)

        train_net(model, opts)

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
