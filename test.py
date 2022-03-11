import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import option
from dataset import Dataset
from sklearn.metrics import roc_curve,roc_auc_score,auc
import numpy as np
from radam import RAdam
import torch.nn.functional as F
from network.video_classifier import  LSTMclassifier,SelfAttentionClassfier,SelfAttentionClassfier_nolstm,SelfAttentionClassfier_nolstm_spational,VideoClassifier
from train import split_cal_auc_videoclassifier

if __name__=='__main__':
    args = option.parser.parse_args()
    seed = args.seed
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    da_list = [args.da]
    r_list = [args.r]

    test_data = Dataset(args,test_mode=True,is_normal=False,is_onmemory=True)
    test_dataloader = DataLoader(test_data,batch_size=1,shuffle=False)

    aucs = [[] for da in da_list]
    for i,da in enumerate(da_list):
        for r in r_list:
            model = VideoClassifier(args,r=r,da=da)
            model_name = model_name = 'SelfAttention-da{}-r{}'.format(da,r)
            model.load_state_dict(torch.load('save_weight/{}-T{}-seed{}.pth'.format(model_name,args.T,args.seed)))
            auc_score,_,_ = split_cal_auc_videoclassifier(model,test_dataloader,device,split_size=args.test_split_size)
            aucs[i].append(auc_score)
            print('model : {} , auc_score : {}'.format(model_name,auc_score))
    
    np.save('list/video_classifier_parameter_test.npy',np.array(aucs))