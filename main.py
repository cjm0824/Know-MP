# -*- coding: utf-8 -*-
import time
from model import knowMP
from train import evaluation, train_model, test_model
import torch
import random   
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


model_name = 'know-MP'
encoder_name = 'BERT'

dataset2config = {"FewRel":  {"taskname":"Relation Extraction",
                               "meta_lr": 5e-5,
                               "task_lr": 1e-2,
                               "weight decay": 1e-2,#权重衰减系数
                               "batch_size": 32,
                               "train_iters": 1000,
                               "steps": 30,
                               "max_length":90,
                               "warmup_step":200
                               },
                    }

benchmark = "FewRel"
taskname = dataset2config[benchmark]['taskname']
meta_lr = dataset2config[benchmark]['meta_lr']
task_lr  = dataset2config[benchmark]['task_lr']
weight_decay = dataset2config[benchmark]['weight decay']
B = dataset2config[benchmark]['batch_size']
Train_iter = dataset2config[benchmark]['train_iters']
Fast_tuning_steps = dataset2config[benchmark]['steps']
max_length = dataset2config[benchmark]['max_length']
warmup_step = dataset2config[benchmark]['warmup_step']

noise_rate = 0 #  from 0 to 10

N = 5
K = 5
Q = 1

Val_iter = 2000
Val_step = 50

save_ckpt = f'./checkpoint/{benchmark}_know-MP_{N}-way {K}-shot noisr:{noise_rate}.pth'
load_ckpt = None
best_acc = 0.0



print('----------------------------------------------------')
print("{}-way-{}-shot Few-Shot {}, dataset is {}, noise_rate = {}%".format(N, K,taskname, benchmark, noise_rate/10*100))
print("Model: {}".format(model_name))
print("Encoder: {}".format(encoder_name))
print('----------------------------------------------------')

if benchmark=="FewRel":
    data_dir = {'benchmark': benchmark,
                'train': f'FewRel2/train.json',
                'val': f'FewRel2/val_pubmed_type.json',
                'test': f'FewRel2/val_pubmed_type.json',
                'noise_rate': noise_rate,
                'candidates': f'FewRel2/merged_file.json',
                'pb_dropout': 0.5}
else:
    data_dir = {'benchmark': benchmark,
                'train':f'./data/{benchmark}/train.json',
                'val':f'./data/{benchmark}/val.json',
                'test':f'./data/{benchmark}/test.json',
                'noise_rate': noise_rate,
                'candidates': f'./data/{benchmark}/candidate_ebds.json',
                'pb_dropout': 0.5}
                               
start_time=time.time()

knowMP=knowMP(B,N,K,max_length,data_dir)

# train_model(knowMP,B,N,K,Q,data_dir,
#             meta_lr=meta_lr,
#             task_lr=task_lr,
#             weight_decay = weight_decay,
#             train_iter=Train_iter,
#             val_iter=Val_iter,
#             val_step=Val_step,
#             steps=Fast_tuning_steps,
#             save_ckpt=save_ckpt, load_ckpt= load_ckpt,
#             best_acc=best_acc,
#             warmup_step=warmup_step
#             )

load_ckpt = f'./checkpoint/{benchmark}_know-MP_{N}-way {K}-shot noisr_{noise_rate}.pth'
# evaluation(knowMP,N,K,Q,eval_iter=2000, steps=Fast_tuning_steps,task_lr=task_lr, noise_rate = 0,file_name=f'FewRel2/val_pubmed_type.json',load_ckpt=load_ckpt)
test_model(knowMP,N,K,Q,eval_iter=10000, steps=Fast_tuning_steps,task_lr=task_lr, noise_rate = 0,file_name=f'FewRel2/test-{N}-{K}_type.json',load_ckpt=load_ckpt)

time_use=time.time()-start_time
h=int(time_use/3600)
time_use-=h*3600
m=int(time_use/60)
time_use-=m*60
s=int(time_use)
print('Totally used',h,'hours',m,'minutes',s,'seconds')
