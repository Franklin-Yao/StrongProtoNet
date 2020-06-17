import os
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.datamgr import SimpleDataManager, SetDataManager
from utils import Averager, Timer, count_acc,compute_confidence_interval, Model_type, Method_type,\
    save_model, load_pretrained_weights, init, resume_model


def train_one_epoch(model, optimizer, args, train_loader, label, writer, epoch):
    model.train()

    print_freq = 10
    for i, batch in enumerate(train_loader):
        data, index_label = batch[0].cuda(), batch[1].cuda()

        all_logits = model(data, 'train')
        if args.method_type is Method_type.baseline:
            label = index_label
        loss = F.cross_entropy(all_logits, label)
        acc = count_acc(all_logits, label)
        if i % print_freq == print_freq - 1:
            if args.exp_tag in ['same_labels']:
                print('epoch {}, train {}/{}, loss={:.4f}, KL_loss={:.4f}, acc={:.4f}'.format(epoch, i,
                                                                                              len(train_loader),
                                                                                              loss.item(), loss1.item(),
                                                                                              acc))
            else:
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), loss.item(), acc))

        if writer is not None:
            writer.add_scalar('loss', loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def val(model, args, val_loader, label):
    model.eval()
    vl = Averager()
    va = Averager()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader, 1), total=len(val_loader)):
            data, index_label = batch[0].cuda(), batch[1]
            logits = model(data, mode='val')
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

    vl = vl.item()
    va = va.item()
    return vl, va


def test(model, label, args, few_shot_params):
    if args.debug:
        n_test = 10
        print_freq = 2
    else:
        n_test = 1000
        print_freq = 100
    test_file = args.dataset_dir + 'test.json'
    test_datamgr = SetDataManager(test_file, args.dataset_dir, args.image_size,
                                  mode = 'val',n_episode = n_test ,**few_shot_params)
    loader = test_datamgr.get_data_loader(aug=False)

    test_acc_record = np.zeros((n_test,))

    warmup_state = torch.load(osp.join(args.checkpoint_dir, 'max_acc' + '.pth'))['params']
    model.load_state_dict(warmup_state, strict=False)
    model.eval()

    ave_acc = Averager()
    with torch.no_grad():
        for i, batch in enumerate(loader, 1):
            data, index_label = batch[0].cuda(), batch[1].cuda()
            logits = model(data, 'test')
            acc = count_acc(logits, label)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            if i % print_freq == 0:
                print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    m, pm = compute_confidence_interval(test_acc_record)
    # print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'],
    #                                                               ave_acc.item()))
    print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
    acc_str = '%4.2f' % (m * 100)
    with open(args.save_dir + '/result.txt', 'a') as f:
        f.write('%s %s\n' % (acc_str, args.name))

def main():
    timer = Timer()
    args, writer = init()

    train_file = args.dataset_dir + 'train.json'
    val_file = args.dataset_dir + 'val.json'

    few_shot_params = dict(n_way=args.n_way, n_support=args.n_shot, n_query=args.n_query)
    n_episode = 10 if args.debug else 100
    if args.method_type is Method_type.baseline:
        train_datamgr = SimpleDataManager(train_file, args.dataset_dir, args.image_size, batch_size=64)
        train_loader = train_datamgr.get_data_loader(aug = True)
    else:
        train_datamgr = SetDataManager(train_file, args.dataset_dir, args.image_size,
                                       n_episode=n_episode, mode='train', **few_shot_params)
        train_loader = train_datamgr.get_data_loader(aug=True)

    val_datamgr = SetDataManager(val_file, args.dataset_dir, args.image_size,
                                     n_episode=n_episode, mode='val', **few_shot_params)
    val_loader = val_datamgr.get_data_loader(aug=False)

    if args.model_type is Model_type.ConvNet:
        pass
    elif args.model_type is Model_type.ResNet12:
        from methods.backbone import ResNet12
        encoder = ResNet12()
    else:
        raise ValueError('')

    if args.method_type is Method_type.baseline:
        from methods.baselinetrain import BaselineTrain
        model = BaselineTrain(encoder, args)
    elif args.method_type is Method_type.protonet:
        from methods.protonet import ProtoNet
        model = ProtoNet(encoder, args)
    else:
        raise ValueError('')

    from torch.optim import SGD,lr_scheduler
    if args.method_type is Method_type.baseline:
        optimizer = SGD(model.encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=0, last_epoch=-1)
    else:
        optimizer = torch.optim.SGD(model.encoder.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4,
                                    nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    args.ngpu = torch.cuda.device_count()
    torch.backends.cudnn.benchmark = True
    model = model.cuda()

    label = torch.from_numpy(np.repeat(range(args.n_way), args.n_query))
    label = label.cuda()

    if args.test:
        test(model, label, args, few_shot_params)
        return

    if args.resume:
        resume_OK =  resume_model(model, optimizer, args, scheduler)
    else:
        resume_OK = False
    if (not resume_OK) and  (args.warmup is not None):
        load_pretrained_weights(model, args)

    if args.debug:
        args.max_epoch = args.start_epoch + 1

    for epoch in range(args.start_epoch, args.max_epoch):
        train_one_epoch(model, optimizer, args, train_loader, label, writer, epoch)
        scheduler.step()

        vl, va = val(model, args, val_loader, label)
        if writer is not None:
            writer.add_scalar('data/val_acc', float(va), epoch)
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va >= args.max_acc:
            args.max_acc = va
            print('saving the best model! acc={:.4f}'.format(va))
            save_model(model, optimizer, args, epoch, args.max_acc, 'max_acc', scheduler)
        save_model(model, optimizer, args, epoch, args.max_acc, 'epoch-last', scheduler)
        if epoch != 0:
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    if writer is not None:
        writer.close()
    test(model, label, args, few_shot_params)

if __name__ == '__main__':
    main()
