import argparse, random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

CHARS='0123456789ABEKMHOPCTYX'
FONTS=['/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','C:/Windows/Fonts/arialbd.ttf','C:/Windows/Fonts/arial.ttf']

def get_font(px):
    for p in FONTS:
        if Path(p).exists(): return ImageFont.truetype(p,px)
    return ImageFont.load_default()

class GlyphDataset(Dataset):
    def __init__(self,n=12000,seed=20261025): self.n=n; self.seed=seed
    def __len__(self): return self.n
    def __getitem__(self,i):
        rng=random.Random(self.seed+i)
        y=rng.randrange(len(CHARS)); ch=CHARS[y]
        im=Image.new('L',(40,56),rng.randint(225,255)); d=ImageDraw.Draw(im)
        f=get_font(rng.randint(34,47)); b=d.textbbox((0,0),ch,font=f)
        x=(40-(b[2]-b[0]))//2-b[0]+rng.randint(-3,3); yy=(56-(b[3]-b[1]))//2-b[1]+rng.randint(-3,3)
        d.text((x,yy),ch,font=f,fill=rng.randint(0,35))
        if rng.random()<.7: im=im.rotate(rng.uniform(-7,7),resample=Image.Resampling.BILINEAR,fillcolor=255)
        if rng.random()<.6: im=im.filter(ImageFilter.GaussianBlur(rng.uniform(.1,1.0)))
        a=np.asarray(im,dtype=np.float32)/255.0
        noise=np.random.default_rng(self.seed+i).normal(0,rng.uniform(0,.06),a.shape).astype(np.float32)
        a=np.clip(a+noise,0,1)
        a=1-a
        return torch.from_numpy(a[None]), y

class OCRNet(nn.Module):
    def __init__(self):
        super().__init__(); self.features=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,3)))
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*4*3,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,len(CHARS)))
    def forward(self,x): return self.head(self.features(x))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=4); ap.add_argument('--samples',type=int,default=12000); ap.add_argument('--out',default='models/ocr_cnn.pt'); a=ap.parse_args()
    torch.manual_seed(20261025); ds=GlyphDataset(a.samples); dl=DataLoader(ds,batch_size=128,shuffle=True,num_workers=0)
    m=OCRNet(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3); lossf=nn.CrossEntropyLoss()
    for e in range(a.epochs):
        m.train(); n=ok=0; tot=0
        for x,y in dl:
            opt.zero_grad(); z=m(x); loss=lossf(z,y); loss.backward(); opt.step(); tot+=float(loss)*len(y); ok+=(z.argmax(1)==y).sum().item(); n+=len(y)
        print(f'epoch {e+1}: loss={tot/n:.4f} acc={ok/n:.4f}')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':m.state_dict(),'chars':CHARS,'input_hw':[56,40],'seed':20261025,'training':'procedural glyphs'},a.out)
    print('saved',a.out)
if __name__=='__main__': main()
