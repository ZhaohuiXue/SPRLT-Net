# -*- coding: utf-8 -*-
"""
@author:
"""
import torch
import numpy as np
import os
import sklearn
import scipy.io as sio
from data import dataset_size_dict, data_name_dict, dataset_class_dict, color_map_dict
from torch.utils.data import Dataset, DataLoader, TensorDataset


def get_device():
    # Use GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def print_results(data_name, oa, aa, kappa, class_acc, traintime, testtime):
    # run_times and runtime
    # output the results into a txt file and a mat file.
    n_class = dataset_class_dict[data_name]
    mean_oa = format(np.mean(oa * 100), '.2f')
    std_oa = format(np.std(oa * 100), '.2f')
    mean_aa = format(np.mean(aa) * 100, '.2f')
    std_aa = format(np.std(aa) * 100, '.2f')
    mean_kappa = format(np.mean(kappa) * 100, '.2f')
    std_kappa = format(np.std(kappa) * 100, '.2f')

    print('\n')
    print('train_time:', str(np.mean(traintime)), 'std:', str(np.std(traintime)))
    print('test_time:', str(np.mean(testtime)), 'std:', str(np.std(testtime)))
    
    
    for i in range(n_class):
        mean_std = str(round(np.mean(class_acc[:, i]) * 100, 2)) + '±' + str(round(np.std(class_acc[:, i]) * 100, 2))
        print('Class ', str(i + 1), ' mean ± std:', mean_std)
    
    print('OA mean:', str(mean_oa), 'std:', str(std_oa))
    print('AA mean:', str(mean_aa), 'std:', str(std_aa))
    print('Kappa mean:', str(mean_kappa), 'std:', str(std_kappa))


def draw_allresult(labels, dataset_name='PU', border=False):
    num_class = np.max(labels) + 1
    row = dataset_size_dict[str(dataset_name)][0]
    col = dataset_size_dict[str(dataset_name)][1]
    # sio.savemat('./map.mat', {'x': np.reshape(labels, (row, col))})
    palette = color_map_dict[str(dataset_name)]
    palette = palette * 1.0 / 255
    X_result = np.zeros((row * col, 3))
    
    for i in range(0, num_class):
        X_result[np.where(labels == i), 0] = palette[i, 0]
        X_result[np.where(labels == i), 1] = palette[i, 1]
        X_result[np.where(labels == i), 2] = palette[i, 2]
    
    X_result = np.reshape(X_result, (row, col, 3))
    if border:
        new_X_result = np.zeros([row + 2, col + 2, 3])
        new_X_result[1:-1, 1:-1, :] = X_result
        X_result = new_X_result
    newlab=np.zeros(row*col)
    newlab[:labels.shape[0]]=labels
    return X_result, np.reshape(newlab, (row, col))


def draw_labelresult(labels, index, dataset_name='PU', border=True):
    num_class = np.max(labels) + 1
    row = dataset_size_dict[str(dataset_name)][0]
    col = dataset_size_dict[str(dataset_name)][1]
    # sio.savemat('./map.mat', {'x': np.reshape(labels, (row, col))})
    
    palette = color_map_dict[str(dataset_name)]
    palette = palette * 1.0 / 255
    lab = np.ones((row * col)) * (-1)
    lab[index] = labels
    X_result = np.zeros((row * col, 3))
    
    for i in range(0, num_class):
        X_result[np.where(lab == i), 0] = palette[i, 0]
        X_result[np.where(lab == i), 1] = palette[i, 1]
        X_result[np.where(lab == i), 2] = palette[i, 2]
    
    X_result[np.where(lab == -1), 0] = 255 * 1.0 / 255
    X_result[np.where(lab == -1), 1] = 255 * 1.0 / 255
    X_result[np.where(lab == -1), 2] = 255 * 1.0 / 255
    
    X_result = np.reshape(X_result, (row, col, 3))
    if border:
        new_X_result = np.zeros([row + 2, col + 2, 3])
        new_X_result[1:-1, 1:-1, :] = X_result
        X_result = new_X_result
    
    return X_result, labels


def sampling(ground_truth, train_proportion=0.1, train_list=[], seed=666):
    random_state = np.random.RandomState(seed=seed)
    train = {}
    val = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)[0]
    for i in range(m):
        indexes = [
            j for j, x in enumerate(ground_truth.ravel().tolist())
            if x == i + 1
        ]
        random_state.shuffle(indexes)
        
        labels_loc[i] = indexes
        if train_list:
            nb_val = train_list[i]
        else:
            if train_proportion != 1:
                nb_val = max(int((train_proportion) * len(indexes)), 5)
            else:
                nb_val = 0
        train[i] = indexes[:int(nb_val*1)]
        val[i] = indexes[int(nb_val*1):nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    val_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        val_indexes += val[i]
        test_indexes += test[i]
    random_state.shuffle(train_indexes)
    random_state.shuffle(val_indexes)
    random_state.shuffle(test_indexes)
    
    train_idx = np.array(train_indexes)
    test_idx = np.array(test_indexes)
    val_idx = np.array(val_indexes)
    drawlabel_idx = np.array(train_indexes + test_indexes)
    drawall_idx = np.array([j for j, x in enumerate(ground_truth.ravel().tolist())])
    return train_idx, test_idx, val_idx, drawlabel_idx, drawall_idx


def sampling_disjoint(ground_truth):
    Y_train = ground_truth[0]
    Y_test = ground_truth[1]
    n_class = Y_test.max()
    train_idx = list()
    test_idx = list()
    val_idx = list()
    for i in range(1, n_class + 1):
        train_i = np.where(Y_train == i)[0]
        test_i = np.where(Y_test == i)[0]
        
        train_idx.extend(train_i[:int(len(train_i)*1)])
        val_idx.extend(train_i[int(len(train_i)*1):])
        test_idx.extend(test_i)
    
    drawlabel_idx = np.array(train_idx + val_idx + test_idx)
    drawall_idx = np.array([j for j, x in enumerate(Y_train)])
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    return train_idx, test_idx, val_idx, drawlabel_idx, drawall_idx


'''
especially for spectralformer
'''
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band

def generate_batch(idx, X_PCAMirrow, Y, batch_size, ws, dataset_name, shuffle=False):
    num = len(idx)
    hw = ws // 2
    row = dataset_size_dict[dataset_name][0]
    col = dataset_size_dict[dataset_name][1]
    
    if shuffle:
        np.random.shuffle(idx)

    for i in range(0, num, batch_size):
        # if num-i<batch_size:
        #     continue
        bi = np.array(idx)[np.arange(i, min(num, i + batch_size))]
        index_row = np.ceil((bi + 1) * 1.0 / col).astype(np.int32)
        index_col = (bi + 1) - (index_row - 1) * col
        # index_row += hw - 1
        # index_col += hw - 1
        patches = np.zeros([bi.size, ws, ws, X_PCAMirrow.shape[-1]])
        for j in range(bi.size):
            a = index_row[j] - 1#hw
            b = index_col[j] - 1#hw
            patch = X_PCAMirrow[a:a + ws, b:b + ws, :]  # *np.reshape(np.repeat(sa_lab[:,:,bi[j]],200),(9,9,200))
            patches[j, :, :, :] = patch
        
        # patches = np.array(patches)#.reshape([batch_size,ws,ws,patch.shape[-1]])
        labels = Y[bi, :] - 1
        # patches=gain_neighborhood_band(x_train=patches,band=X_PCAMirrow.shape[2],band_patch=3,patch=ws)
        
        yield patches, labels[:,0] # torch.nn.functional.one_hot(torch.Tensor(labels[:,0]).to(torch.int64), Y.max()).float()


if __name__ == "__main__":
    pass


