import torch.utils.data as data
import numpy as np
import torch
import option
from tqdm import tqdm

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False,is_onmemory=False,getvideolabel=False):
        
        self.is_normal = is_normal
        if test_mode:
            self.rgb_list_file = args.test_rgb_list
        else:
            self.rgb_list_file = args.pre_train_ts_list
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None
        self.is_onmemory = is_onmemory
        self.getvideolabel = getvideolabel
        if is_onmemory == True:
            self._get_onmemory()

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.is_normal:
                self.list = self.list[810:]
                #print('normal list')
                #print(self.list)
            else:
                self.list = self.list[:810]
                #print('abnormal list')
                #print(self.list)
    
    def _get_onmemory(self):
        dataset_features = []
        for list_i in tqdm(self.list):
            features = np.load(list_i.strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            dataset_features.append(features)
        
        self.dataset_features = dataset_features

    def __getitem__(self, index):
        
        label = self.get_label(index) # get video level label 0/1
        if self.is_onmemory == True:
            features = self.dataset_features[index]
            if self.test_mode:
                if self.getvideolabel:
                    return features,self.list[index].strip('\n')
                else:
                    return features
            else:
                return features, label
        else:
            features = np.load(self.list[index].strip('\n'), allow_pickle=True)
            features = np.array(features, dtype=np.float32)

            if self.tranform is not None:
                features = self.tranform(features)
            if self.test_mode:
                return features
            else:
                return features, label
        

    def get_label(self, index):
        if self.is_normal:
            # label[0] = 1
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
            # label[1] = 1
        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame