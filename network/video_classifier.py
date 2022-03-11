import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class LSTMclassifier(nn.Module):
    def __init__(self,args):
        super(LSTMclassifier,self).__init__()
        self.bilstm = nn.LSTM(args.feature_size,256,batch_first=True,bidirectional=True,num_layers=2)
        self.fc = nn.Linear(256*2,1)
        self.dropout = nn.Dropout(p=0.5)
        self.sig = nn.Sigmoid()
        
        nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self,x):
        bs,t,f = x.shape
        #x = x.view(-1,t,f) #[batch,32,2048]
        out,(hidden,_) = self.bilstm(x)
        x = out[:,-1,:] #[batch,256*2]
        x = self.dropout(x)
        x = self.sig(self.fc(x)) #[batch,1]
        x = x.view(bs) 
        
        return x


class SelfAttentionClassfier(nn.Module):
    def __init__(self,args,lstm_dim=256):
        super(SelfAttentionClassfier,self).__init__()
        self.bilstm = nn.LSTM(args.feature_size,lstm_dim,batch_first=True,bidirectional=True,num_layers=2)
        self.self_anttention = nn.Sequential(nn.Linear(lstm_dim*2,64,bias=True),nn.Tanh(),nn.Linear(64,3,bias=True))
        self.fc1 = nn.Linear(lstm_dim*6,32)
        self.fc2 = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        bs,t,f = x.shape
        out,_ = self.bilstm(x) #out.shape = [batch,32,256]
        attentionweight = self.dropout(F.softmax(self.self_anttention(out),dim=1))
        m1 = (out*attentionweight[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (out*attentionweight[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (out*attentionweight[:,:,2].unsqueeze(2)).sum(dim=1)
        x = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = x.view(bs) #[batch]

        return x

class SelfAttentionClassfier_nolstm(nn.Module):
    def __init__(self,args,lstm_dim=256,attention=False):
        super().__init__()
        #self.bilstm = nn.LSTM(args.feature_size,lstm_dim)
        self.self_anttention = nn.Sequential(nn.Linear(args.feature_size,64),nn.Tanh(),nn.Linear(64,3))
        self.fc1 = nn.Linear(args.feature_size*3,32)
        self.fc2 = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        self.return_attention = attention

    def forward(self,x):
        bs,t,f = x.shape
        #out,_ = self.bilstm(x) #out.shape = [batch,32,256]
        out = x
        attentionweight = self.dropout(F.softmax(self.self_anttention(out),dim=1))
        m1 = (out*attentionweight[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (out*attentionweight[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (out*attentionweight[:,:,2].unsqueeze(2)).sum(dim=1)
        x = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = x.view(bs) #[batch]

        if self.return_attention:
            return x, attentionweight
        else:
            return x


class SelfAttentionClassfier_nolstm_spational(nn.Module):
    def __init__(self,args,attention=False):
        super().__init__()
        #self.bilstm = nn.LSTM(args.feature_size,lstm_dim)
        self.self_anttention = nn.Sequential(nn.Linear(args.feature_size,64),nn.Tanh(),nn.Linear(64,3))
        self.fc1 = nn.Linear(args.feature_size*3,32)
        self.fc2 = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        self.return_attention = attention
        self.softmax = nn.Softmax2d()

    def forward(self,x):
        bs,t,f = x.shape
        #out,_ = self.bilstm(x) #out.shape = [batch,32,256]
        out = x
        attentionweight = self.dropout(self.softmax(self.self_anttention(out)))
        m1 = (out*attentionweight[:,:,0].unsqueeze(2)).sum(dim=1)
        m2 = (out*attentionweight[:,:,1].unsqueeze(2)).sum(dim=1)
        m3 = (out*attentionweight[:,:,2].unsqueeze(2)).sum(dim=1)
        x = torch.cat([m1,m2,m3],dim=1) #[batch*10,128*6]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = x.view(bs) #[batch]

        if self.return_attention:
            return x, attentionweight
        else:
            return x

class VideoClassifier(nn.Module):
    def __init__(self,args,r=3,da=64,attention=False):
        super().__init__()
        self.self_anttention = nn.Sequential(nn.Linear(args.feature_size,da),nn.Tanh(),nn.Linear(da,r))
        self.fc1 = nn.Linear(args.feature_size*r,32)
        self.fc2 = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()

        self.return_attention = attention

    def forward(self,x):
        bs,t,f = x.shape        
        attentionweight = self.dropout(F.softmax(self.self_anttention(x),dim=1))
        m = torch.bmm(x.permute(0,2,1),attentionweight)
        x = m.view(bs,-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = x.view(bs) #[batch]

        if self.return_attention:
            return x, attentionweight
        else:
            return x