# -*- coding: utf-8 -*-
"""
@author:
"""
from train_test import loop_train_test
from data import dataset_size_dict
import warnings
import time

# remove abundant output
warnings.filterwarnings('ignore')

## global constant value
run_times = 10  # 20
output_map = False
only_draw_label = False  # draw full map or labeled map
disjoint = True

model_type = 'SPRLT'  # Model: {'RNN','D2CNN', 'SSRN','DBMA','A2S2K','Transformer','VIT','SSFTT','SpectralFormer','HPDM_SPRN', 'DFFN'}
ws = 9
epochs = 30
batch_size = 64
lr = 5e-4

def pavia_university_experiment():
    train_proportion = 0.01
    num_list = []
    pcadimension = 0
    hp = {
        'dataset': 'PU',
        'run_times': run_times,
        'pchannel': dataset_size_dict['PU'][2] if pcadimension == 0 else pcadimension,
        'model': model_type,
        'ws': ws,
        'epochs': 30,#30,25
        'batch_size': batch_size,
        'learning_rate': lr,
        'train_proportion': train_proportion,
        'train_num': num_list,
        'outputmap': output_map,
        'only_draw_label': only_draw_label,
        'disjoint': True
    }
    loop_train_test(hp)


if __name__ == '__main__':
    pavia_university_experiment()
    print(time.asctime(time.localtime()))
