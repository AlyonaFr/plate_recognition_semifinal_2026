import argparse, csv, random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

LETTERS='ABEKMHOPCTYX'; DIGITS='0123456789'


def font(px):
    for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf','C:/Windows/Fonts/arialbd.ttf','C:/Windows/Fonts/arial.ttf']:
        if Path(p).exists(): return ImageFont.truetype(p,px)
    return ImageFont.load_default()

def plate_num(rng):
    region = str(rng.randint(10,99)) if rng.random()<0.45 else rng.choice('127')+f'{rng.randint(0,99):02d}'
    return rng.choice(LETTERS)+''.join(rng.choice(DIGITS) for _ in range(3))+rng.choice(LETTERS)+rng.choice(LETTERS)+region

def draw_text_center(draw, box, text, f, fill):
    b=draw.textbbox((0,0),text,font=f); tw=b[2]-b[0]; th=b[3]-b[1]
    x=box[0]+(box[2]-box[0]-tw)//2; y=box[1]+(box[3]-box[1]-th)//2-b[1]
    draw.text((x,y),text,font=f,fill=fill)

def plate_image(num, typ, rng):
    if typ=='type1a':
        W,H=290,170; bg=(245,245,240); fg=(12,12,12)
        im=Image.new('RGB',(W,H),bg); d=ImageDraw.Draw(im); d.rectangle((3,3,W-4,H-4),outline=fg,width=4)
        draw_text_center(d,(8,8,W-8,H//2+5),num[:4],font(68),fg)
        draw_text_center(d,(8,H//2-3,W-8,H-8),num[4:],font(52),fg)
    else:
        W,H=520,112; bg=(245,205,35) if typ=='type1b' else (245,245,240); fg=(12,12,12)
        im=Image.new('RGB',(W,H),bg); d=ImageDraw.Draw(im); d.rectangle((3,3,W-4,H-4),outline=fg,width=4)
        draw_text_center(d,(10,5,W-10,H-5),num,font(62),fg)
    return im

def augment(im,rng):
    arr=np.array(im)
    # perspective warp
    h,w=arr.shape[:2]; m=max(2,int(min(h,w)*0.08))
    src=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst=src+np.float32([[rng.randint(-m,m),rng.randint(-m,m)] for _ in range(4)])
    M=cv2.getPerspectiveTransform(src,dst)
    arr=cv2.warpPerspective(arr,M,(w,h),borderValue=(90,90,90))
    if rng.random()<0.6:
        k=rng.choice([3,5]); arr=cv2.GaussianBlur(arr,(k,k),rng.uniform(0.2,1.2))
    if rng.random()<0.7:
        noise=np.random.default_rng(rng.randint(0,2**31-1)).normal(0,rng.uniform(2,10),arr.shape).astype(np.float32)
        arr=np.clip(arr.astype(np.float32)+noise,0,255).astype(np.uint8)
    alpha=rng.uniform(0.72,1.18); beta=rng.randint(-22,22); arr=cv2.convertScaleAbs(arr,alpha=alpha,beta=beta)
    return arr

def compose(plate,rng):
    H,W=360,640
    base=np.zeros((H,W,3),np.uint8)
    c=rng.randint(60,180); base[:]=[c+rng.randint(-20,20),c+rng.randint(-20,20),c+rng.randint(-20,20)]
    # car-like horizontal bands
    cv2.rectangle(base,(0,90),(W,300),(rng.randint(30,130),rng.randint(30,130),rng.randint(30,130)),-1)
    ph,pw=plate.shape[:2]
    scale=rng.uniform(0.45,0.95); nw=max(40,int(pw*scale)); nh=max(20,int(ph*scale))
    plate=cv2.resize(plate,(nw,nh),interpolation=cv2.INTER_AREA)
    x=rng.randint(10,max(10,W-nw-10)); y=rng.randint(120,max(120,H-nh-20))
    base[y:y+nh,x:x+nw]=plate
    # final camera degradation
    if rng.random()<0.45: base=cv2.GaussianBlur(base,(3,3),0.5)
    return base,(x,y,nw,nh)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--count',type=int,default=5000); ap.add_argument('--seed',type=int,default=20261025); ap.add_argument('--root',default='dataset'); a=ap.parse_args()
    root=Path(a.root); out=root/'images'/'synthetic'; labels=root/'labels'; out.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True)
    meta=root/'meta.csv'; exists=meta.exists() and meta.stat().st_size>0
    rng=random.Random(a.seed)
    with meta.open('a',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,delimiter=';')
        if not exists: wr.writerow(['image','plate_num','plate_type','bbox','quad','is_vehicle','is_synthetic','source','license','conditions'])
        for i in range(a.count):
            typ=rng.choices(['type1','type1a','type1b'],weights=[0.34,0.33,0.33])[0]; num=plate_num(rng)
            p=np.array(plate_image(num,typ,rng)); p=augment(Image.fromarray(p),rng); img,b=compose(p,rng)
            name=f'syn_{a.seed}_{i:06d}.jpg'; rel=f'images/synthetic/{name}'; cv2.imwrite(str(out/name),img,[cv2.IMWRITE_JPEG_QUALITY,rng.randint(72,94)])
            x,y,w,h=b; quad=f'{x},{y},{x+w},{y},{x+w},{y+h},{x},{y+h}'
            wr.writerow([rel,num,typ,f'{x},{y},{w},{h}',quad,1,1,'generator','CC BY 4.0','day,angle'])
            (labels/(Path(name).stem+'.txt')).write_text(f'{typ};{num};{x},{y},{w},{h};{quad}\n',encoding='utf-8')
    print(f'Generated {a.count} images into {out}')
if __name__=='__main__': main()
