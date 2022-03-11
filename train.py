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
from network.video_classifier import VideoClassifier

class Sampling_replace_iter:
    def __init__(self,features,n_sample,is_normal=True,replace=True,T_shuffle=False):

        self.features = np.array(features)
        self.is_normal = is_normal
        self.n_sample = n_sample
        self.indexs = np.arange(0,len(features))
        np.random.shuffle(self.indexs)
        self.count = 0
        self.replace = replace
        self.T_shuffle = T_shuffle
        
    def __iter__(self):
        
        if self.is_normal:
            labels = torch.zeros(self.n_sample).long()
        else:
            labels = torch.ones(self.n_sample).long()
        self.count = 0

        # print(self.features.shape) [8000,T,2048]
        if self.T_shuffle:
            bs,t,f = self.features.shape
            self.features = self.features[:][np.random.shuffle(np.arange(t))]
            self.features = self.features.reshape(bs,t,f)

        if not self.replace:
            np.random.shuffle(self.indexs)

        while self.count+self.n_sample < len(self.indexs):
            if self.replace:
                choice_index = np.random.choice(self.indexs,self.n_sample,replace=True).tolist()
                yield torch.tensor(self.features[choice_index]) ,labels
                self.count += self.n_sample
            else:
                choice_index = self.indexs[self.count:self.count+self.n_sample].tolist()
                yield torch.tensor(self.features[choice_index]),labels
                self.count += self.n_sample


def split_cal_auc_mil(model,dataloader,device,split_size=32,gt_path="list/gt-ucf.npy"):
    model.to(device)
    model.eval()
    bag_pred_scores = []

    with torch.no_grad():
        for feature in tqdm(dataloader):
            #print('feature.shape = ',feature.shape) #[1,58,10,2048]
            feature = feature.permute(0,2,1,3)
            feature = feature.squeeze(0)
            split_feature = torch.split(feature,split_size,dim=1)
            for feature_i in split_feature:
                feature_i = feature_i.to(device)
                out = model(feature_i)
                out = out.mean(0).reshape(-1).detach().cpu()
                #out = out.repeat(feature_i.size(1))
                bag_pred_scores.append(out)
            
    gt = np.load(gt_path)
    bag_pred_scores = torch.cat(bag_pred_scores).numpy()
    pred_score = np.repeat(bag_pred_scores,16)
    #print('gt.shape = ',gt.shape) #[1114144] for UCF-Crime
    #print('pred_score.shape = ',pred_score.shape) #[1114144] for UCF-Crime
    fpr, tpr, thresholds = roc_curve(gt,pred_score)
    auc_score = auc(fpr,tpr)
    
    return auc_score,fpr,tpr

def split_cal_auc_videoclassifier(model,dataloader,device,split_size=32,gt_path="list/gt-ucf.npy"):
    model.to(device)
    model.eval()
    bag_pred_scores = []

    with torch.no_grad():
        for feature in tqdm(dataloader):
            #print('feature.shape = ',feature.shape) #[1,58,10,2048]
            feature = feature.permute(0,2,1,3)
            feature = feature.squeeze(0)
            split_feature = torch.split(feature,split_size,dim=1)
            for feature_i in split_feature:
                feature_i = feature_i.to(device)
                out = model(feature_i)
                out = out.mean(0).reshape(-1).detach().cpu()
                out = out.repeat(feature_i.size(1))
                bag_pred_scores.append(out)
            
    gt = np.load(gt_path)
    bag_pred_scores = torch.cat(bag_pred_scores).numpy()
    pred_score = np.repeat(bag_pred_scores,16)
    #print('gt.shape = ',gt.shape) #[1114144] for UCF-Crime
    #print('pred_score.shape = ',pred_score.shape) #[1114144] for UCF-Crime
    fpr, tpr, thresholds = roc_curve(gt,pred_score)
    auc_score = auc(fpr,tpr)
    
    return auc_score,fpr,tpr

if __name__=='__main__':

    save_weight = True 
    args = option.parser.parse_args()
    seed = args.seed
    torch.manual_seed(seed) 
    np.random.seed(seed)

    da = args.da
    r = args.r

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Use : {}".format(device))
    model_name = 'SelfAttention-da{}-r{}'.format(da,r)
    model = VideoClassifier(args,r=r,da=da)
    model = model.to(device)
    print(model)

    criterion = nn.BCELoss()
    optimizer = RAdam(model.parameters())
    test_data = Dataset(args,test_mode=True,is_normal=False,is_onmemory=True)
    test_dataloader = DataLoader(test_data,batch_size=1,shuffle=False)
    max_auc_score,_,_ =split_cal_auc_videoclassifier(model,test_dataloader,device,split_size=args.test_split_size)
    print('first_auc_score : ',max_auc_score)

    feature_list = list(open(args.pre_rgb_list))
    normal_list = feature_list[810:]
    abnormal_list = feature_list[:810]
    normal_features = []
    abnormal_features = []
    
    for normal_i in tqdm(normal_list):
        feature = np.array(np.load(normal_i.strip('\n'), allow_pickle=True),dtype=np.float32)
        #print(feature.shape)
        feature_split = np.split(feature,10,axis=0)
        feature_split = [split_i.reshape(args.T,2048) for split_i in feature_split]
        for split_i in feature_split:
            normal_features.append(split_i)
    print(len(normal_features))
    
    for abnormal_i in tqdm(abnormal_list):
        feature = np.array(np.load(abnormal_i.strip('\n'), allow_pickle=True),dtype=np.float32)
        feature_split = np.split(feature,10,axis=0)
        feature_split = [split_i.reshape(args.T,2048) for split_i in feature_split]
        for split_i in feature_split:
            abnormal_features.append(split_i)
    
    normal_iter = Sampling_replace_iter(features=normal_features,n_sample=args.batch_size,is_normal=True,replace=False,T_shuffle=False)
    abnormal_iter = Sampling_replace_iter(features=abnormal_features,n_sample=args.batch_size,is_normal=False,replace=False,T_shuffle=False)

    auc_list = [max_auc_score]

    for epoch_i in range(args.max_epoch):
        print('epoch {} '.format(epoch_i))
        model.train()

        for (normal_feature,normal_label),(anomaly_feature,anomaly_label) in zip(normal_iter,abnormal_iter):
            input_datas = torch.cat((normal_feature,anomaly_feature),0)
            input_labels = torch.cat((normal_label,anomaly_label),0).float()
            #print(input_labels.shape) #[batch_size*2]
            #print(input_datas.shape) #[bathc_size,10,32,2048]
            input_datas = input_datas.to(device)
            input_labels = input_labels.to(device)
            output = model(input_datas)
            loss = criterion(output,input_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            auc_score,_,_ = split_cal_auc_videoclassifier(model,test_dataloader,device,split_size=args.test_split_size)
            if max_auc_score < auc_score:
                max_auc_score = auc_score
                if save_weight:
                    torch.save(model.to('cpu').state_dict(),'save_weight/{}-T{}-seed{}.pth'.format(model_name,args.T,args.seed))
                    model = model.to(device)
            print('{}-seed{} max_auc {} auc {}'.format(model_name,args.seed,max_auc_score,auc_score))
            auc_list.append(auc_score)
            model.train()

    #auc_list = np.array(auc_list)
    #np.save('list/{}-seed{}.npy'.format(model_name,args.seed),auc_list)