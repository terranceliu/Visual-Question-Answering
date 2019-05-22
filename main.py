import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import pdb

from preprocess import preprocess
from dataset import VQADataset, VQABatchSampler, CIFAR100Dataset, Corpus
from train import train_model, train_model_lg, test_model
# from vqa_mutan_bilstm import VQAModel as VQAModel
from vqa import VQAModel
from san import SANModel
from scheduler import CustomReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/config_vqa_sgd.yml')


def load_datasets(config, phases):
    num_users = config['num_users']
    config = config['data']
    if 'preprocess' in config and config['preprocess']:
        print('Preprocessing datasets')
        preprocess(
            data_dir=config['dir'],
            train_ques_file=config['train']['ques'],
            train_ans_file=config['train']['ans'],
            val_ques_file=config['val']['ques'],
            val_ans_file=config['val']['ans'])

    print('Loading preprocessed datasets')
    datafiles = {x: '{}.pkl'.format(x) for x in phases}
    raw_images = not ('preprocess' in config['images'] and config['images']['preprocess'])
    if raw_images:
        img_dir = {x: config[x]['img_dir'] for x in phases}
    else:
        img_dir = {x: config[x]['emb_dir'] for x in phases}

    print("Building datasets")
    datasets = {(x, y): VQADataset(data_dir=config['dir'], qafile=datafiles[x], img_dir=img_dir[x], phase=x,
                                   img_scale=config['images']['scale'], img_crop=config['images']['crop'], raw_images=raw_images,
                                   user=y, num_users=num_users)
                for x in phases for y in range(num_users)}

    print("Building batch samplers")
    batch_samplers = {(x, y): VQABatchSampler(datasets[(x, y)], config[x]['batch_size'])
                      for x in phases for y in range(num_users)}

    print("Building dataloaders")
    dataloaders = {(x, y): DataLoader(datasets[(x, y)], batch_sampler=batch_samplers[(x, y)], num_workers=config['loader']['workers'])
                   for x in phases for y in range(num_users)}

    dataset_sizes = {(x, y): len(datasets[(x, y)]) for x in phases for y in range(num_users)}

    print(dataset_sizes)
    print("ques vocab size: {}".format(len(VQADataset.ques_vocab)))
    print("ans vocab size: {}".format(len(VQADataset.ans_vocab)))

    return dataloaders, VQADataset.ques_vocab, VQADataset.ans_vocab


def load_datasets_cifar_helper(config, train=True, download=True, num_users=1):
    config = config['data_cifar']

    if train:
        phase = 'train'
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(config['images']['scale']),
            transforms.CenterCrop(config['images']['crop']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
    else:
        phase = 'val'
        transform = transforms.Compose([
            transforms.Resize(config['images']['scale']),
            transforms.CenterCrop(config['images']['crop']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

    dataloader = CIFAR100(root=config['dir'], train=train, download=download, transform=transform)

    num_examples = len(dataloader)
    indices = np.random.permutation(num_examples)
    indices = np.array_split(indices, num_users)

    datasets = {}
    for i in range(num_users):
        if train:
            datasets[i] = CIFAR100Dataset(dataloader, indices[i])
        else:
            datasets[i] = dataloader

    dataloaders = {}
    for i in range(num_users):
        dataloaders[i] = DataLoader(datasets[i], batch_size=config[phase]['batch_size'], shuffle=True,
                                    num_workers=config['loader']['workers'])

    return dataloaders


def load_datasets_cifar(config, download=True):
    num_users = config['num_users']
    dataloaders_train = load_datasets_cifar_helper(config, train=True, download=download, num_users=num_users)
    dataloaders_val = load_datasets_cifar_helper(config, train=False, download=download, num_users=num_users)

    dataloaders = {}
    dataloaders['train'] = dataloaders_train
    dataloaders['val'] = dataloaders_val

    return dataloaders


def batchify(config, data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if config['use_gpu']:
        data = data.cuda()

    return data


def split_data(data, num_users=1):
    chunk_size = len(data) // num_users
    splits = torch.split(data, chunk_size)
    return splits


def load_datasets_lm(config, ques_dict):
    num_users = config['num_users']
    corpus = Corpus(config['data_lm']['dir'], dictionary=ques_dict)
    vocab = corpus.get_word2ix()

    train_data = split_data(corpus.train, num_users=num_users)
    val_data = corpus.val
    val_data = batchify(config, val_data, config['data_lm']['val']['batch_size'])

    dataloaders = {'train': {}, 'val': {}}
    for i in range(num_users):
        dataloaders['train'][i] = batchify(config, train_data[i], config['data_lm']['train']['batch_size'])
        dataloaders['val'][i] = val_data

    return dataloaders, vocab

def main(config):
    if config['mode'] == 'test':
        phases = ['train', 'test']
    else:
        phases = ['train', 'val']

    num_users = config['num_users']
    local_ep = config['local_ep']
    frac = config['frac']

    dataloaders, ques_vocab, ans_vocab = load_datasets(config, phases)
    dataloaders_cifar = load_datasets_cifar(config)
    dataloaders_lm, vocab = load_datasets_lm(config, ques_vocab)

    # pdb.set_trace()

    # add model parameters to config
    if config['multitask']:
        print("total vocab size: {}".format(len(vocab)))
        config['model']['params']['vocab_size'] = len(vocab)
    else:
        config['model']['params']['vocab_size'] = len(ques_vocab)
    config['model']['params']['output_size'] = len(ans_vocab) - 1   # -1 as don't want model to predict '<unk>'
    config['model']['params']['output_size_lm'] = len(vocab) - 1
    config['model']['params']['extract_img_features'] = 'preprocess' in config['data']['images'] and config['data']['images']['preprocess']
    # which features dir? test, train or validate?
    config['model']['params']['features_dir'] = os.path.join(
         config['data']['dir'], config['data']['train']['emb_dir'])
    if config['model']['type'] == 'vqa':
        model = VQAModel(mode=config['mode'], **config['model']['params'])
    elif config['model']['type'] == 'san':
        model = SANModel(mode=config['mode'], **config['model']['params'])
    print(model)
    criterion = nn.CrossEntropyLoss()

    if config['optim']['class'] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              **config['optim']['params'])
    elif config['optim']['class'] == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                  **config['optim']['params'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               **config['optim']['params'])

    best_acc = 0
    startEpoch = 0
    if 'reload' in config['model']:
        pathForTrainedModel = os.path.join(config['save_dir'],
                                           config['model']['reload'])
        if os.path.exists(pathForTrainedModel):
            print(
                "=> loading checkpoint/model found at '{0}'".format(pathForTrainedModel))
            checkpoint = torch.load(pathForTrainedModel)
            startEpoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
    if config['use_gpu']:
        model = model.cuda()

    print('config mode ', config['mode'])
    save_dir = os.path.join(os.getcwd(), config['save_dir'])

    if config['mode'] == 'train':
        if 'scheduler' in config['optim'] and config['optim']['scheduler'].lower() == 'CustomReduceLROnPlateau'.lower():
            print('CustomReduceLROnPlateau')
            exp_lr_scheduler = CustomReduceLROnPlateau(
                optimizer, config['optim']['scheduler_params']['maxPatienceToStopTraining'], config['optim']['scheduler_params']['base_class_params'])
        else:
            # Decay LR by a factor of gamma every step_size epochs
            print('lr_scheduler.StepLR')
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        print("begin training, local_globaL: {}, multitask: {}, num_users: {}, frac: {}, local_ep: {}". format(
            config['local_global'], config['multitask'], num_users, frac, local_ep))
        if config['local_global']:
            model = train_model_lg(model, dataloaders, dataloaders_cifar, dataloaders_lm, criterion, optimizer, exp_lr_scheduler, config, save_dir,
                            num_epochs=config['optim']['n_epochs'], use_gpu=config['use_gpu'], best_accuracy=best_acc,
                            start_epoch=startEpoch, num_users=num_users, frac=frac, local_ep=local_ep)
        else:
            model = train_model(model, dataloaders, dataloaders_cifar, dataloaders_lm, criterion, optimizer, exp_lr_scheduler, config, save_dir,
                            num_epochs=config['optim']['n_epochs'], use_gpu=config['use_gpu'], best_accuracy=best_acc,
                            start_epoch=startEpoch, num_users=num_users, frac=frac, local_ep=local_ep)

    # elif config['mode'] == 'test':
    #     outputfile = os.path.join(save_dir, config['mode'] + ".json")
    #     test_model(model, dataloaders['test'], VQADataset.ans_vocab,
    #                outputfile, use_gpu=config['use_gpu'])
    else:
        print("Invalid config mode %s !!" % config['mode'])


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    args.config = os.path.join(os.getcwd(), args.config)
    config = yaml.load(open(args.config))
    config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()

    # TODO: seeding still not perfect
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    main(config)
