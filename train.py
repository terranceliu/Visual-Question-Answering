import shutil
import time
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
from IPython.core.debugger import Pdb
from scheduler import CustomReduceLROnPlateau
import json
import numpy as np
import copy
import torch.optim as optim
from collections import defaultdict

import pdb

def train(model, dataloader, criterion, optimizer, use_gpu=False, local_ep=1):
    model.train()  # Set model to training mode

    for epoch in range(local_ep):
        running_loss = 0.0
        running_corrects = 0
        example_count = 0
        step = 0

        # Iterate over data.
        for questions, images, image_ids, answers, ques_ids in dataloader:
            # print('questions size: ', questions.size())
            if use_gpu:
                questions, images, image_ids, answers = questions.cuda(), images.cuda(), image_ids.cuda(), answers.cuda()
            questions, images, answers = Variable(questions).transpose(0, 1), Variable(images), Variable(answers)

            # zero grad
            optimizer.zero_grad()
            ans_scores = model(images, questions, image_ids)
            _, preds = torch.max(ans_scores, 1)
            loss = criterion(ans_scores, answers)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum((preds == answers).data)
            example_count += answers.size(0)
            step += 1
            if step % 1000 == 0:
                print('step {}, running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
                    step, running_loss / example_count, running_corrects, example_count, (float(running_corrects) / example_count) * 100))

            # if step > 3:
            #     break

            batch_size = 100
            decay_rate = 0.3 ** (1 / (50000 * 500 / batch_size))
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * decay_rate  # 0.99997592083

        loss = running_loss / example_count
        acc = float(running_corrects) / example_count * 100 # (running_corrects / len(dataloader.dataset)) * 100
        print('Local Epoch: {} Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(epoch+1, loss, acc, running_corrects, example_count))

    return loss, acc


def validate(model, dataloader, criterion, use_gpu=False, return_probs=False):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    step = 0

    probs = []

    # Iterate over data.
    for questions, images, image_ids, answers, ques_ids in dataloader:
        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(
            ), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(
            0, 1), Variable(images), Variable(answers)

        # zero grad
        ans_scores = model(images, questions, image_ids)
        probs.append(ans_scores.detach())

        _, preds = torch.max(ans_scores, 1)
        loss = criterion(ans_scores, answers)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)

        # step += 1
        # if step > 300:
        #     break

    loss = running_loss / example_count
    # acc = (running_corrects / example_count) * 100
    acc = float(running_corrects) / example_count * 100 # (running_corrects / len(dataloader.dataset)) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                                acc, running_corrects, example_count))

    if return_probs:
        return loss, acc, torch.cat(probs)
    return loss, acc


def train_cifar(model, dataloader, criterion, optimizer, use_gpu=False, local_ep=1):
    model.train()  # Set model to training mode

    for epoch in range(local_ep):
        running_loss = 0.0
        running_corrects = 0
        example_count = 0
        step = 0

        # Iterate over data.
        for images, labels in dataloader:
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # zero grad
            optimizer.zero_grad()
            preds_scores = model(images, None, None, task='cifar100')
            _, preds = torch.max(preds_scores, 1)
            loss = criterion(preds_scores, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_corrects += torch.sum((preds == labels).data)
            example_count += labels.size(0)
            step += 1
            if step % 200 == 0:
                print('step {}, running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
                    step, running_loss / example_count, running_corrects, example_count, (float(running_corrects) / example_count) * 100))
            #
            # if step > 200:
            #     break

        loss = running_loss / example_count
        acc = float(running_corrects) / example_count * 100 # (running_corrects / len(dataloader.dataset)) * 100
        print('Local Epoch: {} Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(epoch+1, loss, acc, running_corrects, example_count))

    return loss, acc


def validate_cifar(model, dataloader, criterion, use_gpu=False):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    step = 0
    # Iterate over data.

    for images, labels in dataloader:
        if use_gpu:
            images, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)

        # zero grad
        preds_scores = model(images, None, None, task='cifar100')
        _, preds = torch.max(preds_scores, 1)
        loss = criterion(preds_scores, labels)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum((preds == labels).data)
        example_count += labels.size(0)

    loss = running_loss / example_count
    # acc = (running_corrects / example_count) * 100
    acc = float(running_corrects) / example_count * 100 # (running_corrects / len(dataloader.dataset)) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                                acc, running_corrects, example_count))
    return loss, acc


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, seq_len=35):
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def train_lm(model, train_data, criterion, optimizer, ntokens, batch_size=1, seq_len=35, clip=0.25, local_ep=1):
    # Turn on training mode which enables dropout.
    model.train()

    for epoch in range(local_ep):
        running_loss = 0.0
        running_corrects = 0
        example_count = 0

        # hidden = model.init_hidden(batch_size)
        for step, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
            data, targets = get_batch(train_data, i, seq_len=seq_len)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # hidden = repackage_hidden(hidden)

            # model.zero_grad()
            output = model(None, data, None, task='lm')
            _, preds = output.max(dim=-1)
            preds = preds.view(-1)

            loss = criterion(output.view(-1, ntokens), targets) * len(data)
            loss.backward()
            optimizer.step()

            # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)

            running_loss += loss.item()
            running_corrects += torch.sum((preds == targets).data)
            example_count += targets.size(0)

            loss = running_loss / (step + 1 * seq_len)
            acc = float(running_corrects) / example_count * 100

            if step % 100 == 0:
                print('step {}, running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
                    step, loss, running_corrects, example_count, acc))

        loss = running_loss / (train_data.size(0) - 1)
        acc = float(running_corrects) / example_count * 100
        print('Local Epoch: {} Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(epoch + 1, loss, acc, running_corrects,
                                                                               example_count))

    return loss, acc


def validate_lm(model, val_data, criterion, ntokens, batch_size=1, seq_len=35):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    example_count = 0

    for step, i in enumerate(range(0, val_data.size(0) - 1, seq_len)):
        data, targets = get_batch(val_data, i, seq_len=seq_len)

        output = model(None, data, None, task='lm')
        _, preds = output.max(dim=-1)
        preds = preds.view(-1)
        loss = criterion(output.view(-1, ntokens), targets) * len(data)

        running_loss += loss.item()
        running_corrects += torch.sum((preds == targets).data)
        example_count += targets.size(0)

    loss = running_loss / (val_data.size(0) - 1)
    acc = float(running_corrects) / example_count * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss, acc, running_corrects, example_count))

    return loss, acc


def train_model(model, data_loaders, data_loaders_cifar, data_loaders_lm, criterion, optimizer, scheduler, config, save_dir,
                num_epochs=25, use_gpu=False, best_accuracy=0, start_epoch=0, num_users=1, frac=0.1, local_ep=1):
    tasks = ['cifar100', 'lm', 'vqa']
    num_tasks = len(tasks)
    ntokens = config['model']['params']['output_size_lm']

    print('Training Model with use_gpu={}...'.format(use_gpu))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = best_accuracy
    writer = SummaryWriter(save_dir)

    w_locals = []
    w_glob = None

    m = max(int(frac * num_users), 1)
    results = []

    for epoch in range(start_epoch, num_epochs):
        if config['multitask']:
            if epoch < config['optim']['pretrain']:
                task_ix = np.random.randint(num_tasks - 1)
            else:
                task_ix = np.random.randint(num_tasks)
        else:
            task_ix = -1
        task = tasks[task_ix]

        print('Epoch {}/{} - Task: {}, lr: {}'.format(epoch, num_epochs - 1, task, optimizer.param_groups[0]['lr']))
        print('-' * 10)

        idxs_users = np.random.choice(range(num_users), m, replace=False)
        grads_local = []
        square_avgs = []

        for user in idxs_users:
            print('User {}'.format(user))
            model_local = copy.deepcopy(model)
            w_keys_epoch = model.state_dict().keys()

            assert len(optimizer.param_groups) == 1
            optimizer_local = copy.deepcopy(optimizer)
            optimizer_local.param_groups[0]['params'] = list(model_local.parameters())

            train_begin = time.time()
            if task == 'cifar100':
                train_loss, train_acc = train_cifar(model_local, data_loaders_cifar['train'][user], criterion, optimizer_local,
                                                    use_gpu, local_ep=local_ep)
            elif task == 'lm':
                train_loss, train_acc = train_lm(model_local, data_loaders_lm['train'][user], criterion, optimizer_local, ntokens, local_ep=local_ep)
            else: # vqa
                train_loss, train_acc = train(model_local, data_loaders[('train', user)], criterion, optimizer_local,
                                              use_gpu, local_ep=local_ep)

            train_time = time.time() - train_begin
            print('Epoch Train Time: {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Train Accuracy', train_acc, epoch)

            if not config['use_grad_norm']:
                grads = 1.0
            else:
                grads = []
                for grad in [param.grad for param in model_local.parameters()]:
                    if grad is not None:
                        grads.append(grad.view(-1))
                grads = torch.cat(grads).norm().item()
                print(grads)
            grads_local.append(grads)

            w_curr = model_local.state_dict()
            if w_glob is None:
                w_glob = w_curr
                for k in w_keys_epoch:
                    w_glob[k] *= grads
            else:
                for k in w_keys_epoch:
                    w_glob[k] += w_curr[k] * grads

            if config['optim']['class'] == 'rmsprop':
                if len(square_avgs) == 0:
                    for p in optimizer_local.param_groups[0]['params']:
                        if p.grad is None:
                            continue
                        square_avgs.append(optimizer_local.state[p]['square_avg'] * grads)
                else:
                    i = 0
                    for p in optimizer_local.param_groups[0]['params']:
                        if p.grad is None:
                            continue
                        square_avgs[i] += optimizer_local.state[p]['square_avg'] * grads
                        i += 1

        for k in w_keys_epoch:
            w_glob[k] = torch.div(w_glob[k], sum(grads_local))

        for i in range(len(square_avgs)):
            square_avgs[i] = torch.div(square_avgs[i], sum(grads_local))

        # copy weight to net_glob
        model.load_state_dict(w_glob)
        optimizer.param_groups[0]['params'] = list(model.parameters())

        if config['optim']['class'] == 'rmsprop':
            optimizer.state = defaultdict(dict)
            i = 0
            for p in optimizer.param_groups[0]['params']:
                if p.grad is None:
                    continue
                optimizer.state[p]['square_avgs'] = square_avgs[i]
                optimizer.state[p]['step'] = (epoch + 1) * config['data']['train']['batch_size']
                i += 1

            optimizer.param_groups[0]['lr'] = optimizer_local.param_groups[0]['lr']

        if epoch < 15:
            continue

        if task=='cifar100':
            validation_begin = time.time()
            val_loss, val_acc = validate_cifar(model, data_loaders_cifar['val'][user], criterion, use_gpu)

            validation_time = time.time() - validation_begin
            print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch)
        elif task == 'lm':
            validation_begin = time.time()
            val_loss, val_acc = validate_lm(model, data_loaders_lm['val'][user], criterion, ntokens)

            validation_time = time.time() - validation_begin
            print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch)
        else:
            validation_begin = time.time()

            val_loss, val_acc = [], []
            for user in range(num_users):
                a, b = validate(model, data_loaders[('val', user)], criterion, use_gpu)
                val_loss.append(a)
                val_acc.append(b)
            val_loss = sum(val_loss) / float(len(val_loss))
            val_acc = sum(val_acc) / float(len(val_acc))

            validation_time = time.time() - validation_begin
            print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch)

            results.append(np.array([epoch, val_loss, val_acc]))  # , val_loss_avg, val_acc_avg
            final_results = np.array(results)
            results_save_path = './results/fed_users{}.npy'.format(
                num_users)
            np.save(results_save_path, final_results)

            # deep copy the model
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                best_model_wts = model.state_dict()

            save_checkpoint(save_dir, {
                'epoch': epoch,
                'best_acc': best_acc,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer.state_dict(),
            }, is_best)

            if is_best:
                savepath = 'best_fed_models/fed_epoch{}_acc{}'.format(epoch, round(best_acc))
                torch.save(model.state_dict(), savepath)

            writer.export_scalars_to_json(save_dir + "/all_scalars.json")
            valid_error = 1.0 - val_acc / 100.0
            if type(scheduler) == CustomReduceLROnPlateau:
                scheduler.step(valid_error, epoch=epoch)
                if scheduler.shouldStopTraining():
                    print("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(
                        scheduler.unconstrainedBadEpochs, scheduler.maxPatienceToStopTraining))
                    # Pdb().set_trace()
                    break
            else:
                scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json(save_dir + "/all_scalars.json")
    writer.close()

    return model


def train_model_lg(model, data_loaders, data_loaders_cifar, data_loaders_lm, criterion, optimizer, scheduler, config, save_dir,
                num_epochs=25, use_gpu=False, best_accuracy=0, start_epoch=0, num_users=1, frac=0.1, local_ep=1):
    num_param_glob = 0
    num_param_local = 0
    for key in model.state_dict().keys():
        if 'extractor' in key:
            continue
        if key in ['mlp_cifar.0.weight', 'mlp_cifar.0.bias', 'mlp_lm.0.weight', 'mlp_lm.0.bias']:
            continue
        num_param_local += model.state_dict()[key].numel()
        if key in model.w_keys:
            num_param_glob += model.state_dict()[key].numel()

        print('{}: {}'.format(key, model.state_dict()[key].numel()))
    percentage_param = 100 * float(num_param_glob) / num_param_local
    print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
        num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))

    # pdb.set_trace()

    if config['load_fed']:
        fed_model_path = config['load_fed_path']
        model.load_state_dict(torch.load(fed_model_path))

    for user_ix in range(num_users):
        model_save_path = 'save/user{}.pt'.format(user_ix)
        torch.save(model.state_dict(), model_save_path)

    tasks = ['cifar100', 'lm', 'vqa']
    num_tasks = len(tasks)
    ntokens = config['model']['params']['output_size_lm']

    print('Training Model with use_gpu={}...'.format(use_gpu))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = best_accuracy
    writer = SummaryWriter(save_dir)

    w_locals = []
    w_glob = None
    w_keys_epoch = model.w_keys
    results = []

    m = max(int(frac * num_users), 1)

    for epoch in range(start_epoch, num_epochs):
        if config['multitask']:
            if epoch < config['optim']['pretrain']:
                task_ix = np.random.randint(num_tasks - 1)
            else:
                task_ix = np.random.randint(num_tasks)
        else:
            task_ix = -1
        task = tasks[task_ix]

        print('Epoch {}/{} - Task: {}, lr: {}'.format(epoch, num_epochs - 1, task, optimizer.param_groups[0]['lr']))
        print('-' * 10)

        idxs_users = np.random.choice(range(num_users), m, replace=False)
        grads_local = []
        square_avgs = []

        for user in idxs_users:
            print('User {}'.format(user))
            model_save_path = 'save/user{}.pt'.format(user_ix)
            model_local = copy.deepcopy(model)
            model_local.load_state_dict(torch.load(model_save_path))

            assert len(optimizer.param_groups) == 1
            optimizer_local = copy.deepcopy(optimizer)
            optimizer_local.param_groups[0]['params'] = list(model_local.parameters())

            train_begin = time.time()
            if task == 'cifar100':
                train_loss, train_acc = train_cifar(model_local, data_loaders_cifar['train'][user], criterion, optimizer_local,
                                                    use_gpu, local_ep=local_ep)
            elif task == 'lm':
                train_loss, train_acc = train_lm(model_local, data_loaders_lm['train'][user], criterion, optimizer_local, ntokens, local_ep=local_ep)
            else: # vqa
                train_loss, train_acc = train(model_local, data_loaders[('train', user)], criterion, optimizer_local,
                                              use_gpu, local_ep=local_ep)

            train_time = time.time() - train_begin
            print('Epoch Train Time: {:.0f}m {:.0f}s'.format(train_time // 60, train_time % 60))
            writer.add_scalar('Train Loss', train_loss, epoch)
            writer.add_scalar('Train Accuracy', train_acc, epoch)

            torch.save(model_local.state_dict(), model_save_path)

            if not config['use_grad_norm']:
                grads = 1.0
            else:
                grads = []
                for param in model_local.__dict__['_modules']['ques_channel'].fflayer.parameters():
                    grad = param.grad
                    if grad is not None:
                        grads.append(grad.view(-1))

                for param in model_local.__dict__['_modules']['image_channel'].fflayer.parameters():
                    grad = param.grad
                    if grad is not None:
                        grads.append(grad.view(-1))

                for param in model_local.__dict__['_modules']['mlp'].parameters():
                    grad = param.grad
                    if grad is not None:
                        grads.append(grad.view(-1))

                grads = torch.cat(grads).norm().item()
                print(grads)
            grads_local.append(grads)

            w_curr = model_local.state_dict()
            if w_glob is None:
                w_glob = w_curr
                for k in w_keys_epoch:
                    w_glob[k] *= grads
            else:
                for k in w_keys_epoch:
                    w_glob[k] += w_curr[k] * grads

            if config['optim']['class'] == 'rmsprop':
                if len(square_avgs) == 0:
                    for p in optimizer_local.param_groups[0]['params']:
                        if p.grad is None:
                            continue
                        square_avgs.append(optimizer_local.state[p]['square_avg'] * grads)
                else:
                    i = 0
                    for p in optimizer_local.param_groups[0]['params']:
                        if p.grad is None:
                            continue
                        square_avgs[i] += optimizer_local.state[p]['square_avg'] * grads
                        i += 1

        for k in w_keys_epoch:
            w_glob[k] = torch.div(w_glob[k], sum(grads_local))

        for i in range(len(square_avgs)):
            square_avgs[i] = torch.div(square_avgs[i], sum(grads_local))

        # copy weight to net_glob
        model.load_state_dict(w_glob)
        optimizer.param_groups[0]['params'] = list(model.parameters())

        for user in range(num_users):
            model_save_path = 'save/user{}.pt'.format(user)
            w_local = torch.load(model_save_path)
            for k in w_keys_epoch:
                w_local[k] = w_glob[k]

            torch.save(model_local.state_dict(), model_save_path)

        if config['optim']['class'] == 'rmsprop':
            optimizer.state = defaultdict(dict)
            i = 0
            for p in optimizer.param_groups[0]['params']:
                if p.grad is None:
                    continue
                optimizer.state[p]['square_avgs'] = square_avgs[i]
                optimizer.state[p]['step'] = (epoch + 1) * config['data']['train']['batch_size']
                i += 1

            optimizer.param_groups[0]['lr'] = optimizer_local.param_groups[0]['lr']

        # if (epoch + 1) <= 1 / frac:
        #     continue

        # if (epoch + 1) % 3 != 0:
        #     continue

        # test on vqa
        # local models
        val_loss = 0
        val_acc = 0
        for user in range(num_users):
            model_save_path = 'save/user{}.pt'.format(user_ix)
            model_local = copy.deepcopy(model)
            model_local.load_state_dict(torch.load(model_save_path))
            a, b = validate(model_local, data_loaders[('val', user)], criterion, use_gpu=use_gpu)
            val_loss += a
            val_acc += b

        val_loss /= num_users
        val_acc /= num_users
        print('Local: Val Loss: {:.4f} Acc: {:2.3f}'.format(val_loss, val_acc))

        # # glob average model
        # model_temp = copy.deepcopy(model)
        # w_keys_temp = model.state_dict().keys()
        # w_glob_temp = {}
        # for user in range(num_users):
        #     w_local = torch.load(model_save_path)
        #
        #     if len(w_glob_temp) == 0:
        #         w_glob_temp = copy.deepcopy(w_local)
        #     else:
        #         for k in w_keys_temp:
        #             w_glob_temp[k] += w_local[k]
        #
        # for k in w_keys_temp:
        #     w_glob_temp[k] = torch.div(w_glob_temp[k], num_users)
        # model_temp.load_state_dict(w_glob_temp)
        #
        # val_loss_avg = 0
        # val_acc_avg = 0
        # for user in range(num_users): # TODO: num_users
        #     a, b = validate(model_temp, data_loaders[('val', user)], criterion, use_gpu=use_gpu)
        #     val_loss_avg += a
        #     val_acc_avg += b
        # val_loss_avg /= num_users
        # val_acc_avg /= num_users
        #
        # print('Global Avg: Val Loss: {:.4f} Acc: {:2.3f}'.format(val_loss_avg, val_acc_avg))

        results.append(np.array([epoch, val_loss, val_acc])) # , val_loss_avg, val_acc_avg
        final_results = np.array(results)
        results_save_path = './results/lg_users{}.npy'.format(
            num_users)
        np.save(results_save_path, final_results)

        # deep copy the model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        valid_error = 1.0 - val_acc / 100.0
        if type(scheduler) == CustomReduceLROnPlateau:
            scheduler.step(valid_error, epoch=epoch)
            if scheduler.shouldStopTraining():
                print("Stop training as no improvement in accuracy - no of unconstrainedBadEopchs: {0} > {1}".format(
                    scheduler.unconstrainedBadEpochs, scheduler.maxPatienceToStopTraining))
                break
        else:
            scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)

    return model


def save_checkpoint(save_dir, state, is_best):
    savepath = save_dir + '/' + 'checkpoint.pth.tar'
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, save_dir + '/' + 'model_best.pth.tar')


def test_model(model, dataloader, itoa, outputfile, use_gpu=False):
    model.eval()  # Set model to evaluate mode
    example_count = 0
    test_begin = time.time()
    outputs = []

    # Iterate over data.
    for questions, images, image_ids, answers, ques_ids in dataloader:

        if use_gpu:
            questions, images, image_ids, answers = questions.cuda(
            ), images.cuda(), image_ids.cuda(), answers.cuda()
        questions, images, answers = Variable(questions).transpose(
            0, 1), Variable(images), Variable(answers)
        # zero grad
        ans_scores = model(images, questions, image_ids)
        _, preds = torch.max(ans_scores, 1)

        outputs.extend([{'question_id': ques_ids[i], 'answer': itoa[str(
            preds.data[i])]} for i in range(ques_ids.size(0))])

        if example_count % 100 == 0:
            print('(Example Count: {})'.format(example_count))
        # statistics
        example_count += answers.size(0)

    json.dump(outputs, open(outputfile, 'w'))
    print('(Example Count: {})'.format(example_count))
    test_time = time.time() - test_begin
    print('Test Time: {:.0f}m {:.0f}s'.format(test_time // 60, test_time % 60))
