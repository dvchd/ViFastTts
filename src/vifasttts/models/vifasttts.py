import torch
from torch import nn
class ViFastTtsModel(nn.Module):
 def __init__(self,config,vocab_sizes):
  super().__init__();h=config['model']['hidden_size'];d=config['model']['component_dim'];self.emb=nn.ModuleList([nn.Embedding(n,d) for n in vocab_sizes]);self.inp=nn.Linear(d*len(vocab_sizes),h);enc=nn.TransformerEncoderLayer(h,config['model']['attention_heads'],config['model']['ffn_size'],batch_first=True);self.enc=nn.TransformerEncoder(enc,config['model']['encoder_layers']);self.duration=nn.Linear(h,1);self.f0=nn.Linear(h,1);self.energy=nn.Linear(h,1);self.mel=nn.Linear(h,config['mel']['n_mels'])
 def forward(self,x):
  h=self.enc(self.inp(torch.cat([e(x[...,i]) for i,e in enumerate(self.emb)],-1)));return {'mel':self.mel(h),'log_duration':self.duration(h).squeeze(-1),'f0':self.f0(h).squeeze(-1),'energy':self.energy(h).squeeze(-1)}
