import argparse,csv,random
from pathlib import Path
import cv2, numpy as np
from PIL import Image,ImageDraw,ImageFont

def font(px):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf','C:/Windows/Fonts/arialbd.ttf']:
        if Path(p).exists(): return ImageFont.truetype(p,px)
    return ImageFont.load_default()
def center(d,box,t,f,fill):
    b=d.textbbox((0,0),t,font=f); x=box[0]+(box[2]-box[0]-(b[2]-b[0]))//2-b[0]; y=box[1]+(box[3]-box[1]-(b[3]-b[1]))//2-b[1]; d.text((x,y),t,font=f,fill=fill)
def make(r):
    kind=r.choice(['trailer','motorcycle','diplomatic','police','military','foreign'])
    if kind=='motorcycle':
        W,H=220,170; bg=(245,245,240); fg=(10,10,10); txt=f'{r.randint(0,9999):04d}\n{r.choice("ABEKMHOPCTYX")}{r.choice("ABEKMHOPCTYX")}{r.randint(10,99)}'
    elif kind=='diplomatic': W,H=520,112; bg=(180,20,25); fg=(245,245,245); txt=f'{r.randint(1,999):03d}D{r.randint(1,999):03d}{r.randint(10,99)}'
    elif kind=='police': W,H=520,112; bg=(20,70,160); fg=(245,245,245); txt=f'{r.choice("ABEKMHOPCTYX")}{r.randint(0,9999):04d}{r.randint(10,99)}'
    elif kind=='military': W,H=520,112; bg=(15,15,15); fg=(245,245,245); txt=f'{r.randint(0,9999):04d}{r.choice("ABEKMHOPCTYX")}{r.choice("ABEKMHOPCTYX")}{r.randint(10,99)}'
    elif kind=='trailer': W,H=520,112; bg=(245,245,240); fg=(10,10,10); txt=f'{r.choice("ABEKMHOPCTYX")}{r.choice("ABEKMHOPCTYX")}{r.randint(0,9999):04d}{r.randint(10,99)}'
    else: W,H=520,112; bg=(245,245,240); fg=(10,10,10); txt=f'ZZ{r.randint(1000,9999)}EU'
    im=Image.new('RGB',(W,H),bg); d=ImageDraw.Draw(im); d.rectangle((3,3,W-4,H-4),outline=fg,width=4)
    if '\n' in txt:
        a,b=txt.split('\n'); center(d,(5,5,W-5,H//2),a,font(52),fg); center(d,(5,H//2,W-5,H-5),b,font(43),fg)
    else: center(d,(8,5,W-8,H-5),txt,font(55),fg)
    return np.array(im),kind
def compose(p,r):
    H,W=360,640; base=np.full((H,W,3),r.randint(65,175),np.uint8); cv2.rectangle(base,(0,95),(W,310),(r.randint(20,150),r.randint(20,150),r.randint(20,150)),-1)
    ph,pw=p.shape[:2]; sc=r.uniform(.4,.9); nw,nh=max(40,int(pw*sc)),max(25,int(ph*sc)); p=cv2.resize(p,(nw,nh)); x=r.randint(10,max(10,W-nw-10)); y=r.randint(115,max(115,H-nh-15)); base[y:y+nh,x:x+nw]=p; return base,(x,y,nw,nh)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--count',type=int,default=600); ap.add_argument('--seed',type=int,default=20261026); ap.add_argument('--root',default='dataset'); a=ap.parse_args(); r=random.Random(a.seed); root=Path(a.root); out=root/'images'/'synthetic'; labels=root/'labels'; out.mkdir(parents=True,exist_ok=True)
    with (root/'meta.csv').open('a',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,delimiter=';')
        for i in range(a.count):
            p,kind=make(r); img,(x,y,w,h)=compose(p,r); name=f'neg_{a.seed}_{i:06d}.jpg'; cv2.imwrite(str(out/name),img); quad=f'{x},{y},{x+w},{y},{x+w},{y+h},{x},{y+h}'
            wr.writerow([f'images/synthetic/{name}','', 'other',f'{x},{y},{w},{h}',quad,1,1,'generator_other','CC BY 4.0',f'day,{kind}'])
            (labels/(Path(name).stem+'.txt')).write_text(f'other;;{x},{y},{w},{h};{quad}\n',encoding='utf-8')
    print('Generated',a.count,'synthetic negative examples')
if __name__=='__main__': main()
