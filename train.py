import random
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from FocalLoss import FocalLoss
from dataloader import dialogDataset
from model import BugListener

seed = 2021


def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_dialog_loaders(train_address, test_address, pretrained_model, batch_size=32, num_workers=0, pin_memory=False):
    train_set = dialogDataset(train_address + '_train.json', pretrained_model)

    test_set = dialogDataset(test_address + '_test.json', pretrained_model)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=train_set.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             collate_fn=test_set.collate_fn,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, test_loader


def train_model(model, loss_func, dataloader, optimizer, epoch, cuda):
    losses, preds, labels, ids = [], [], [], []

    model.train()

    seed_everything(seed + epoch)
    for data in tqdm(dataloader):
        # clear the grad
        optimizer.zero_grad()

        input_ids, token_type_ids, attention_mask_ids, graph_label = \
            [d.cuda() for d in data[:-4]] if cuda else data[:-4]
        dialog_id, role, graph_edge, seq_len = data[-4:]

        log_prob = model(input_ids, token_type_ids, attention_mask_ids, role, seq_len, graph_edge)
        loss = loss_func(log_prob, graph_label)

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(graph_label.cpu().numpy())
        losses.append(loss.item())
        ids += dialog_id

        # accumulate the grad
        loss.backward()
        # optimizer the parameters
        optimizer.step()

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    assert len(preds) == len(labels) == len(ids)
    error_ids = []
    for vid, target, pred in zip(ids, labels, preds):
        if target != pred:
            error_ids.append(vid)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    graph_pre = round(precision_score(labels, preds) * 100, 2)
    graph_rec = round(recall_score(labels, preds) * 100, 2)
    graph_fscore = round(f1_score(labels, preds) * 100, 2)

    return avg_loss, graph_pre, graph_rec, graph_fscore, error_ids


def evaluate_model(model, loss_func, dataloader, cuda):
    losses, preds, labels, ids = [], [], [], []

    model.eval()

    seed_everything(seed)
    with torch.no_grad():
        for data in tqdm(dataloader):
            input_ids, token_type_ids, attention_mask_ids, graph_label = \
                [d.cuda() for d in data[:-4]] if cuda else data[:-4]
            dialog_id, role, graph_edge, seq_len = data[-4:]

            log_prob = model(input_ids, token_type_ids, attention_mask_ids, role, seq_len, graph_edge)
            loss = loss_func(log_prob, graph_label)

            preds.append(torch.argmax(log_prob, 1).cpu().numpy())
            labels.append(graph_label.cpu().numpy())
            losses.append(loss.item())
            ids += dialog_id

    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    assert len(preds) == len(labels) == len(ids)
    error_ids = []
    for vid, target, pred in zip(ids, labels, preds):
        if target != pred:
            error_ids.append(vid)

    avg_loss = round(np.sum(losses) / len(losses), 4)
    graph_pre = round(precision_score(labels, preds) * 100, 2)
    graph_rec = round(recall_score(labels, preds) * 100, 2)
    graph_fscore = round(f1_score(labels, preds) * 100, 2)

    return avg_loss, graph_pre, graph_rec, graph_fscore, error_ids


class Config(object):
    def __init__(self, project):
        self.cuda = True
        self.train_address = f"./data/augmented/{project}"  # angular, appium, docker, dl4j, gitter, typescript
        self.test_address = f"./data/processed/{project}"
        self.pretrained_model = './../pretrained_model/bert_base_uncased'
        self.D_bert = 768
        self.filter_sizes = [2, 3, 4, 5]
        self.filter_num = 50
        self.D_cnn = 100
        self.D_graph = 64
        self.lr = 1e-4
        self.l2 = 1e-5
        self.batch_size = 32
        self.graph_class_num = 2
        self.dropout = 0.5
        self.epochs = 30


if __name__ == '__main__':
    for project in ['dl4j', 'gitter', 'typescript']:  #'angular', 'appium', 'docker', 'dl4j', 'gitter',
        print("======current_project:", project, "======")
        config = Config(project=project)
        cuda = torch.cuda.is_available() and config.cuda
        if cuda:
            print('Running on GPU')
        else:
            print('Running on CPU')


        seed_everything(seed)
        model = BugListener(config.pretrained_model, config.D_bert, config.filter_sizes, config.filter_num,
                            config.D_cnn, config.D_graph, n_speakers=2, graph_class_num=config.graph_class_num,
                            dropout=config.dropout, ifcuda=cuda)


        # 模型装载至cuda
        if cuda:
            model.cuda()

        graph_loss = FocalLoss(gamma=2)
        # graph_loss = nn.NLLLoss()

        # 冻结bert参数，训练时不更新
        for name, params in model.pretrained_bert.named_parameters():
            params.requires_grad = False

        print(get_parameter_number(model))

        # 过滤掉requires_grad = False的参数
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.l2)

        train_loader, test_loader = get_dialog_loaders(config.train_address, config.test_address, config.pretrained_model,
                                                       batch_size=config.batch_size)

        best_f1 = 0
        for e in range(config.epochs):
            start_time = time.time()

            train_loss, train_pre, train_rec, train_f1, train_error = \
                train_model(model, graph_loss, train_loader, optimizer, e, cuda)

            # test
            test_loss, test_pre, test_rec, test_f1, test_error = \
                evaluate_model(model, graph_loss, test_loader, cuda)

            print('epoch:{},train_loss:{},train_pre:{},train_rec:{},train_f1:{},test_loss:{},test_graph_pre:{},'
                  'test_graph_rec:{},test_graph_f1:{},time:{}sec'.format(e + 1, train_loss, train_pre, train_rec, train_f1,
                                                                         test_loss, test_pre, test_rec, test_f1,
                                                                         round(time.time() - start_time, 2)))
            # print(test_error)
            if test_f1 > best_f1:
                best_f1 = test_f1
            # print("train_error", train_error)
            # print("test_error", test_error)
            print("==========================================================================")

        print("training finish!!!")
        print("best_f1:", best_f1)
