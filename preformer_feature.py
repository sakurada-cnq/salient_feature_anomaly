import numpy as np
import os
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

save_prefeature_path = './preformer_features'
test_list = list(open('./list/ucf-i3d-test-base.list'))
test_base_path = os.path.join(os.getcwd(),'features/UCF-Test')
train_list = list(open('./list/ucf-i3d-train-base.list'))

test_pre_list = []
for test_i in tqdm(test_list):
    test_i = os.path.basename(test_i.strip('\n'))
    test_pre_list.append(os.path.join(test_base_path,test_i)+'\n')    

f = open('list/ucf-i3d-test.list','w')
f.writelines(test_pre_list)
f.close()
if not os.path.isdir(save_prefeature_path):
    os.makedirs(save_prefeature_path)
train_pre_list = []
for train_i in tqdm(train_list):
    if (os.path.exists(train_i.strip('\n')) == False):
        continue
    features = np.load(train_i.strip('\n'),allow_pickle=True)
    features = features.transpose(1, 0, 2)  # [10, B, T, F]
    divided_features = []
    for feature in features:
            feature = process_feat(feature, 32)
            divided_features.append(feature)
    divided_features = np.array(divided_features, dtype=np.float32)
    save_path = os.path.join(save_prefeature_path,os.path.basename(train_i.strip('\n')))
    train_pre_list.append(save_path+'\n')
    np.save(save_path,divided_features)
    #print(divided_features.shape)

f = open('list/pre-ucf-i3d-train-32.list','w')
f.writelines(train_pre_list)
f.close()


