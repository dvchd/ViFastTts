import torch
from torch import nn
class ISTFTNetLite(nn.Module):
 def __init__(self,n_mels=100,n_fft=1024,hop=256):super().__init__();self.n_fft=n_fft;self.hop=hop;self.net=nn.Conv1d(n_mels,(n_fft//2+1)*2,3,padding=1)
 def forward(self,mel):
  z=self.net(mel.transpose(1,2));r,i=z.chunk(2,1);win=torch.hann_window(self.n_fft,device=mel.device);return torch.stack([torch.istft(torch.complex(a,b),self.n_fft,self.hop,self.n_fft,window=win) for a,b in zip(r,i)])
