# -*- coding: utf-8 -*-
"""
@author:
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import time, os
import scipy.io as sio
import collections
import matplotlib.pyplot as plt
from operator import truediv
from sklearn import metrics
from util import sampling, sampling_disjoint, get_device, generate_batch, print_results,draw_labelresult, draw_allresult
from data import lazyprocessing

from network.sprltNet import SPRLT

device = get_device()


def resolve_dict(hp):
    return hp['dataset'], hp['run_times'], hp['pchannel'], hp['model'], hp['ws'], hp['epochs'], \
           hp['batch_size'], hp['learning_rate'], hp['train_proportion'], hp['train_num'], hp['outputmap'], \
           hp['only_draw_label'], hp['disjoint']


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=0)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='sum'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = torch.Tensor(X)
            y = torch.Tensor(y)
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            n += y.shape[0]
    net.train()
    return [acc_sum / n, test_l_sum / test_num]


def train(net, train_idx, val_idx,
          ground_truth, ground_test, X_PCAMirrow,
          batch_size, ws, dataset_name,
          loss, optimizer, device,
          epochs, accumulation_steps=1, early_stopping=True, early_num=10):
    best_acc = 0.5
    net.train()

    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []

    lr_adjust = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    
    for epoch in range(epochs):
        train_lorder = generate_batch(train_idx, X_PCAMirrow, ground_truth, batch_size, ws, dataset_name,
                                      shuffle=True)
        valida_lorder = generate_batch(val_idx, X_PCAMirrow, ground_test, batch_size, ws, dataset_name,
                                       shuffle=True)
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        
        for step, (X, y) in enumerate(train_lorder):
            batch_count, train_l_sum = 0, 0
            X = torch.Tensor(X)  # .type(torch.HalfTensor)
            y = torch.Tensor(y).type(torch.LongTensor)
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X)
            # l = loss(y_hat, y.long())
            # optimizer.zero_grad()
            # l.backward()
            # optimizer.step()
            # 梯度累积
            
            # if (step + 1) % accumulation_steps == 0:
            #     is_step = True
            # else:
            #     is_step = False
            # l = loss(y_hat, y.long()) / accumulation_steps
            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            # if is_step:
            
            optimizer.step()
            
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        
        if (epoch + 1) % 1 == 0:
            # valida_acc, valida_loss = evaluate_accuracy(valida_lorder, net, loss, device)
            valida_loss = 0
            valida_acc = 0
            # pass
        else:
            valida_loss = 0
            valida_acc = 0
        
        train_loss_list.append(train_l_sum / batch_count)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)
        # torch.cuda.memory_summary()
        print('epoch %d, train loss %.5f, train acc %.2f, valida loss %.5f, valida acc %.2f, time %.2f sec, lr: %.6f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n*100,
                 valida_loss, valida_acc*100, time.time() - time_epoch, lr_adjust.get_last_lr()[0]))
        
        # early stop
        # if early_stopping and loss_list[-2] < loss_list[-1]:
        #     if early_epoch == 0 and valida_acc > best_acc:
        #         torch.save(net.state_dict(), "./models/temp_model.pt")
        #         best_acc = valida_acc
        #     early_epoch += 1
        #     loss_list[-1] = loss_list[-2]
        #     if early_epoch == early_num:
        #         net.load_state_dict(torch.load("./models/temp_model.pt"))
        #         break
        # else:
        #     early_epoch = 0
        
        if valida_acc > best_acc:
            best_acc = valida_acc
        torch.save(net.state_dict(), "./models/temp_model.pt")
        lr_adjust.step()
    
    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
             time.time() - start))
        


def test(net, test_loder):
    pred_test = []
    manifold = []
    
    with torch.no_grad():
        for X, _ in test_loder:
            X = torch.Tensor(X)
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
            manifold.extend(np.array(_))
    pred_test = np.array(pred_test)
    manifold = np.array(manifold)
    return pred_test, manifold


def loop_train_test(hyper_parameter):
    datasetname, run_times, num_PC, model_type, w, epochs, batch_size, lr, \
    train_proportion, num_list, outputmap, only_draw_label, disjoint = resolve_dict(hyper_parameter)
    print('>' * 10, "Data set Loading", '<' * 10)
    X_PCAMirrow, ground_truth, shapelist, hsidata = lazyprocessing(datasetname, num_PC=num_PC, w=w, disjoint=disjoint)
    classnum = np.max(ground_truth)
    print(datasetname, 'shape:', shapelist)
    
    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((run_times, classnum))

    print('>' * 10, "Start Training", '<' * 10)
    for run_i in range(0, run_times):
        print('round:', run_i + 1)
        net = SPRLT(
            hidden_dim=128,
            layers=(3),
            heads=(12),
            channels=num_PC,
            num_classes=classnum,
            head_dim=24,
            window_size=5,
            relative_pos_embedding=True
        )
        
        print("load model")
        # net.half()#半精度
        net = net.to(device)
        net.train()

        
        optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)  # 1e-4
        # optimizer = optim.RMSprop(net.parameters(),lr=lr)
        # optimizer = optim.SGD(
        #     net.parameters(),
        #     lr=lr,
        #     weight_decay=0)
        
        loss = torch.nn.CrossEntropyLoss()
        
        if disjoint:
            train_indexes, test_indexes, val_indexes, drawlabel_indexes, drawall_indexes = sampling_disjoint(
                ground_truth)

            ground_truth_train = ground_truth[0]
            ground_truth_test = ground_truth[1]
            print("Training sapmles:", len(train_indexes))
            print("Testing sapmles:", len(test_indexes))
            print("Val sapmles:", len(val_indexes))
            torch.cuda.synchronize()
            tic1 = time.time()
            train(net, train_indexes, test_indexes, ground_truth_train, ground_truth_test, X_PCAMirrow,
                  batch_size, w,
                  datasetname, loss, optimizer, device, epochs)
            torch.cuda.synchronize()
            toc1 = time.time()
        
        else:
            train_indexes, test_indexes, val_indexes, drawlabel_indexes, drawall_indexes = sampling(ground_truth,
                                                                                                    train_proportion,
                                                                                                    num_list,
                                                                                                    seed=(run_i + 1) * 111)
            ground_truth_train = ground_truth
            ground_truth_test = ground_truth
            
            print("Training sapmles:", len(train_indexes))
            print("Testing sapmles:", len(test_indexes))

            print("Val sapmles:", len(val_indexes))
            torch.cuda.synchronize()
            tic1 = time.time()
            train(net, train_indexes, val_indexes, ground_truth_train, ground_truth_test, X_PCAMirrow,
                  batch_size, w,
                  datasetname, loss, optimizer, device, epochs)
            torch.cuda.synchronize()
            toc1 = time.time()
            print('train time:', toc1 - tic1)
        
        test_loder = generate_batch(test_indexes, X_PCAMirrow, ground_truth_test, batch_size*3, w, datasetname,
                                    shuffle=True)
        print('>' * 10, "Start Testing", '<' * 10)
        net.load_state_dict(torch.load("./models/temp_model.pt"))
        
        torch.cuda.synchronize()
        tic2 = time.time()
        pred_test, manifold = test(net, test_loder)
        torch.cuda.synchronize()
        toc2 = time.time()
        
        collections.Counter(pred_test)
        gt_test = manifold
        
        overall_acc = metrics.accuracy_score(pred_test, gt_test[:])
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:])
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, gt_test[:])
        
        if not os.path.exists('./models/{}'.format(model_type)):
            os.makedirs('./models/{}'.format(model_type))

        # net.load_state_dict(torch.load('./models/spatical/vithsi0.948.pt'))
        
        print(confusion_matrix)
        print('OA :', overall_acc)
        print('AA :', average_acc)
        print('Kappa :', kappa)
        print("Each acc:", each_acc)
        print('test time:', toc2 - tic2)
        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[run_i, :] = each_acc
        
        
        if outputmap:
            print('>' * 10, "Start Drawmap", '<' * 10)
            tic3 = time.time()
            if only_draw_label:
                drawmap_loder = generate_batch(drawlabel_indexes, X_PCAMirrow, ground_truth_train, batch_size * 5, w,
                                               datasetname, shuffle=False)  # 画图的时候不要打乱,画图的时候不需要Groundtruch,
                pred_map, _ = test(net, drawmap_loder)
                map, labelmat = draw_labelresult(labels=pred_map, index=drawlabel_indexes, dataset_name=datasetname)
            
            else:
                drawmap_loder = generate_batch(drawall_indexes, X_PCAMirrow, ground_truth_train, batch_size * 10, w,
                                               datasetname, shuffle=False)
                pred_map, _ = test(net, drawmap_loder)
                map, labelmat = draw_allresult(labels=pred_map, dataset_name=datasetname)
            toc3 = time.time()
            print('drawmap time:', toc3 - tic3)
            if not os.path.exists('./classification_maps/{}/{}'.format(model_type, datasetname)):
                os.makedirs('./classification_maps/{}/{}'.format(model_type, datasetname))
            
            # plt.imsave(
            #     './classification_maps/{}/{}/oa_{}.eps'.format(model_type, datasetname, str(int(overall_acc * 10000))),
            #     map)
            
            plt.imsave(
                './classification_maps/{}/{}/oa_{}.png'.format(model_type, datasetname, str(int(overall_acc * 10000))),
                map)
            # plt.imsave(
            #     './classification_maps/{}/{}/oa_{}.svg'.format(model_type, datasetname, str(int(overall_acc * 10000))),
            #     map)
            
            if not os.path.exists('./resmat/{}/{}'.format(model_type, datasetname)):
                os.makedirs('./resmat/{}/{}'.format(model_type, datasetname))
            
            sio.savemat('./resmat/{}/{}/reg_results_{}.mat'.format(model_type, datasetname, run_i + 1),
                        {'OA': overall_acc, 'AA': average_acc, 'Kappa': kappa, 'CA': each_acc,
                         'traintime': toc1 - tic1, 'testtime': toc2 - tic2, 'map': labelmat})

    print_results(datasetname, np.array(OA), np.array(AA), np.array(KAPPA), np.array(ELEMENT_ACC),
                  np.array(TRAINING_TIME), np.array(TESTING_TIME))


