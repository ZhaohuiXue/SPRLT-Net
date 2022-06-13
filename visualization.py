# -*- coding: utf-8 -*-
"""
@author:
"""
import torch

import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL.Image as Image
from data import lazyprocessing, dataset_size_dict,mirror_concatenate
from util import sampling_disjoint,sampling
from network.sprltNet import SPRLT

datasetname = 'HU'
num_PC = 144
w = 39


def generate_batch(idx, X_PCAMirrow,img, Y, batch_size, ws, dataset_name, shuffle=False):
    num = len(idx)
    hw = ws // 2
    row = dataset_size_dict[dataset_name][0]
    col = dataset_size_dict[dataset_name][1]
    gt=Y.reshape(row,col)
    
    if shuffle:
        np.random.shuffle(idx)
    
    for i in range(0, num, batch_size):
        bi = np.array(idx)[np.arange(i, min(num, i + batch_size))]
        index_row = np.ceil((bi + 1) * 1.0 / col).astype(np.int32)
        index_col = (bi + 1) - (index_row - 1) * col


        patches = np.zeros([bi.size, ws, ws, X_PCAMirrow.shape[-1]])
        # pics = np.zeros([bi.size, ws, ws, img.shape[-1]])
        for j in range(bi.size):
            a = index_row[j] - 1
            b = index_col[j] - 1
            patch = X_PCAMirrow[a:a + ws, b:b + ws, :]  # *np.reshape(np.repeat(sa_lab[:,:,bi[j]],200),(9,9,200))
            # pic = img[a:a + ws, b:b + ws, :]
            patches[j, :, :, :] = patch
            # pics[j, :, :, :] = pic
        
        # patches = np.array(patches)#.reshape([batch_size,ws,ws,patch.shape[-1]])
        labels = Y[bi, :] - 1
        aa=index_row[0]-1
        bb=index_col[0]-1
        gt_y=gt[aa,bb]-1
        yield patches, labels[:,0],gt_y,bi,index_row[0],index_col[0]# torch.nn.functional.one_hot(torch.Tensor(labels[:,0]).to(torch.int64), Y.max()).float()

def visual_sprlt():
    img = plt.imread("./datamap/Fig_houston_falsecolor.png")
    img_extension = mirror_concatenate(img)
    
    b = 35 - w // 2
    img_extension = img_extension[b:-b, b:-b, :]
    data_lorder = generate_batch(train_indexes, X_PCAMirrow, img_extension, ground_truth[0], 1, w, datasetname,
                                 shuffle=True)
    
    model = SPRLT(
        hidden_dim=128,  # 96#128
        layers=(3),
        heads=(12),  ##more
        channels=num_PC,
        num_classes=15,
        head_dim=24,
        window_size=13,
        relative_pos_embedding=True
    ).to("cuda")
    
    checkpoint = torch.load("./models/visualSPRLT_HU.pt", map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)
    model.eval()
    
    # print(model)
    # print("^"*10)
    
    all_block = []
    for model_child in list(model.children()):
        if type(model_child) != torch.nn.modules.container.ModuleList:
            all_block.append(model_child)
        else:
            blocks = list(model_child.children())
            for block in blocks:
                all_block.append(block.attention_block)
                all_block.append(block.mlp_block)
    
    print('block_num:', len(all_block))
    
    for step, (x, pic, y) in enumerate(data_lorder):
        # if step > 10:
        #     break
        input_tensor = torch.Tensor(x).to('cuda')
        results = []
        results = [all_block[0](input_tensor)]
        for i in range(1, len(all_block) - 1):
            results.append(all_block[i](results[-1]))
        # make a copy of the `results`
        outputs = []
        for output in results[:-1]:
            outputs.append(output.mean(dim=[3]).cpu())
        
        print(len(outputs))
        # plt.figure(figsize=(7, 35))
        plt.subplot(1, 7, 1)
        plt.imshow(pic.squeeze(0))
        plt.axis("off")
        # plt.imsave("./experiment_res/visual/subpatch/hu_{}_{}.png".format(step, 1), pic.squeeze(0))
        for num_layer in range(0, len(outputs)):
            layer_viz = outputs[num_layer]
            layer_viz = layer_viz.data
            layer_viz = layer_viz.squeeze(0)
            map = layer_viz
            print(layer_viz.shape)
            plt.subplot(1, 7, num_layer + 2)
            # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
            plt.imshow(map, cmap='jet')
            plt.axis("off")
            # plt.imsave("./experiment_res/visual/subpatch/hu_{}_{}.png".format(step,num_layer+2),map)
            
            # layer_viz = layer_viz.reshape(1024, 24, 24)  # change the reshape here
        # plt.show()  # use this line to show the figure in jupyter notebook
        
        plt.savefig("./experiment_res/visual/{}_class_hu{}.png".format(y, step), bbox_inches='tight')
        
        plt.close()
        print('done')

if __name__ == '__main__':

    for w in [1]:
        X_PCAMirrow, ground_truth, shapelist, hsidata = lazyprocessing(datasetname, num_PC=num_PC, w=w, disjoint=False)
        # train_indexes, test_indexes, val_indexes, drawlabel_indexes, drawall_indexes = sampling_disjoint(ground_truth)
        train_indexes, test_indexes, val_indexes, drawlabel_indexes, drawall_indexes = sampling(ground_truth)
        # visual_sprlt()

            
        print(train_indexes.shape)
        data_lorder = generate_batch(train_indexes, X_PCAMirrow, X_PCAMirrow, ground_truth, 1, w, datasetname,
                                     shuffle=True)
        axis=np.array([x for x in range(0,num_PC)])
        spectrals=[]
        x_tr=np.array([x for x in range(1, 145)])
        for step, (x,y,gt_y,idx,rrr,ccc) in enumerate(data_lorder):
            if y[0] == 3:
                value=np.squeeze(x,0).mean(axis=(0,1))
                plt.plot(x_tr,value)
            
                    
                    
        plt.show()






