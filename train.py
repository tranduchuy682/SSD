import time
from argparse import ArgumentParser
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import tqdm
# from model_vgg16 import SSD300, MultiBoxLoss
# from model_mobilenetv3 import SSD300, MultiBoxLoss
# from model_resnet18 import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
parser = ArgumentParser(description='Input backbone')
parser.add_argument('bb', help='Nhap ten backbone')
parser.add_argument('epoch', help='Nhap so epoch')
args = parser.parse_args()
backbone = args.bb
train_epoch = args.epoch


if backbone == "resnet18":
    print('Running backbone ResNet18')
    from model_resnet18 import SSD300, MultiBoxLoss
elif backbone == "mobilenetv3":
    print('Running backbone MobileNetV3')
    from model_mobilenetv3 import SSD300, MultiBoxLoss
else:
    print('Running backbone VGG16')
    backbone = 'vgg16'
    from model_vgg16 import SSD300, MultiBoxLoss



# Data parameters
data_folder = './'  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = '/home/mcn/DucHuy_K63/SSD/SSD-base/weight/checkpoint_ssd300'+backbone+'.pth.tar' #None  # path to model checkpoint, None if none
# checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 2 #4  # number of workers for loading data in the DataLoader
lr = 1e-3  # learning rate
decay_lr_at = [1000*i for i in range(1,120)] #[80000, 100000]  # decay learning rate after these many iterations
decay_lr_to =  0.3 #0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        _loss=[]
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        _loss = checkpoint['_loss']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    # epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, start_epoch+int(train_epoch)):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch, _loss=_loss)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, _loss, backbone)


def train(train_loader, model, criterion, optimizer, epoch, _loss):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    losses = AverageMeter()  # loss


    # Batches
    for (images, boxes, labels, _) in tqdm.tqdm(train_loader):

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
        _loss.append(loss)
        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))


        # Print status
        # if i % len(train_loader) == 0:
    print('Epoch: [{0}]\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
