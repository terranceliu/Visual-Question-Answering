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

        loss = running_loss / example_count
        acc = float(running_corrects) / example_count * 100 # (running_corrects / len(dataloader.dataset)) * 100
        print('Local Epoch: {} Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(epoch+1, loss, acc, running_corrects, example_count))

    return loss, acc


def validate(model, dataloader, criterion, use_gpu=False):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    example_count = 0
    step = 0

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
        loss = criterion(ans_scores, answers)

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum((preds == answers).data)
        example_count += answers.size(0)

        step += 1
        if step > 300:
            break

    loss = running_loss / example_count
    # acc = (running_corrects / example_count) * 100
    acc = float(running_corrects) / example_count * 100 # (running_corrects / len(dataloader.dataset)) * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss,
                                                                acc, running_corrects, example_count))
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
                questions, labels = images.cuda(), labels.cuda()
            images, labels = Variable(images), Variable(labels)

            # zero grad
            optimizer.zero_grad()
            preds_scores = model(images.cuda(), None, None, task='cifar100') #TODO: not sure why I need to call .cuda() again
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
            questions, labels = images.cuda(), labels.cuda()
        images, labels = Variable(images), Variable(labels)

        # zero grad
        preds_scores = model(images.cuda(), None, None, task='cifar100') #TODO: not sure why I need to call .cuda() again
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

        hidden = model.init_hidden(batch_size)
        for step, i in enumerate(range(0, train_data.size(0) - 1, seq_len)):
            data, targets = get_batch(train_data, i)
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)

            model.zero_grad()
            output, hidden = model(None, data, None, hidden=hidden, task='lm')
            _, preds = torch.max(output, 1)


            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()
            optimizer.step()

            # # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)

            running_loss += loss.item()
            running_corrects += torch.sum((preds == targets).data)
            example_count += targets.size(0)


            if step % 1000 == 0:
                print('step {}, running loss: {}, running_corrects: {}, example_count: {}, acc: {}'.format(
                    step, running_loss / example_count, running_corrects, example_count, (float(running_corrects) / example_count) * 100))

        loss = running_loss / example_count
        acc = float(running_corrects) / example_count * 100  # (running_corrects / len(dataloader.dataset)) * 100
        print('Local Epoch: {} Train Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(epoch + 1, loss, acc, running_corrects,
                                                                               example_count))

    return loss, acc


def validate_lm(model, val_data, criterion, ntokens, batch_size=1, seq_len=35):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    example_count = 0

    hidden = model.init_hidden(batch_size)
    for step, i in enumerate(range(0, val_data.size(0) - 1, seq_len)):
        data, targets = get_batch(val_data, i)

        output, hidden = model(None, data, None, hidden=hidden, task='lm')
        _, preds = torch.max(output, 1)
        loss = criterion(output.view(-1, ntokens), targets) # * len(data) ???

        running_loss += loss.item()
        running_corrects += torch.sum((preds == targets).data)
        example_count += targets.size(0)

        hidden = repackage_hidden(hidden)

    loss = running_loss / example_count
    acc = float(running_corrects) / example_count * 100
    print('Validation Loss: {:.4f} Acc: {:2.3f} ({}/{})'.format(loss, acc, running_corrects, example_count))

    return loss, acc


def train_model(model, data_loaders, data_loaders_cifar, data_loaders_lm, criterion, optimizer, scheduler, config, save_dir,
                num_epochs=25, use_gpu=False, best_accuracy=0, start_epoch=0, num_users=1, frac=0.1, local_ep=1):
    tasks = ['vqa', 'cifar100']
    num_tasks = len(tasks)
    ntokens = config['model']['params']['vocab_size']

    print('Training Model with use_gpu={}...'.format(use_gpu))
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = best_accuracy
    writer = SummaryWriter(save_dir)

    w_locals = []
    w_glob = None

    m = max(int(frac * num_users), 1)

    for epoch in range(start_epoch, num_epochs):
        if config['multitask']:
            task_ix = np.random.randint(num_tasks)
        else:
            task_ix = 0
        task = tasks[task_ix]

        print('Epoch {}/{} - Task: {}'.format(epoch, num_epochs - 1, task))
        print('-' * 10)

        idxs_users = np.random.choice(range(num_users), m, replace=False)
        grads_local = []
        square_avgs = []

        for user in idxs_users:
            print('User {}'.format(user))
            model_local = copy.deepcopy(model)

            assert len(optimizer.param_groups) == 1
            optimizer_local = copy.deepcopy(optimizer)
            optimizer_local.param_groups[0]['params'] = list(model_local.parameters())

            # if config['optim']['class'] == 'sgd':
            #     optimizer_local = optim.SGD(filter(lambda p: p.requires_grad, model_local.parameters()),
            #                           **config['optim']['params'])
            # elif config['optim']['class'] == 'rmsprop':
            #     optimizer_local = optim.RMSprop(filter(lambda p: p.requires_grad, model_local.parameters()),
            #                               **config['optim']['params'])
            # else:
            #     optimizer_local = optim.Adam(filter(lambda p: p.requires_grad, model_local.parameters()),
            #                            **config['optim']['params'])

            train_begin = time.time()
            if task == 'cifar100':
                train_loss, train_acc = train_cifar(model_local, data_loaders_cifar['train'][user], criterion, optimizer_local,
                                                    use_gpu, local_ep=local_ep)
            elif task == 'lm':
                train_loss, train_acc = train_lm(model_local, data_loaders_lm['train'][user], criterion, optimizer_local, ntokens,
                                                batch_size=config['data_lm']['val']['batch_size'], local_ep=local_ep)
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
                for k in w_glob.keys():
                    w_glob[k] *= grads
            else:
                for k in w_glob.keys():
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

        for k in w_glob.keys():
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

        # import pdb
        # pdb.set_trace()

        if task=='cifar100':
            validation_begin = time.time()
            val_loss, val_acc = validate_cifar(model, data_loaders_cifar['val'][user], criterion, use_gpu)

            validation_time = time.time() - validation_begin
            print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch)
        elif task == 'lm':
            validation_begin = time.time()
            val_loss, val_acc = validate_lm(model, data_loaders_lm['val'][user], criterion, ntokens,
                                            batch_size=config['data_lm']['val']['batch_size'])

            validation_time = time.time() - validation_begin
            print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch)
        else:
            validation_begin = time.time()
            val_loss, val_acc = validate(model, data_loaders[('val', user)], criterion, use_gpu)

            validation_time = time.time() - validation_begin
            print('Epoch Validation Time: {:.0f}m {:.0f}s'.format(validation_time // 60, validation_time % 60))
            writer.add_scalar('Validation Loss', val_loss, epoch)
            writer.add_scalar('Validation Accuracy', val_acc, epoch)

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
