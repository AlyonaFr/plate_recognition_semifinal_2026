import argparse, random
from pathlib import Path
import cv2, numpy as np, joblib
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
CHARS='0123456789ABEKMHOPCTYX'
FONTS=['/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','C:/Windows/Fonts/arialbd.ttf','C:/Windows/Fonts/arial.ttf']
def font(px):
    for p in FONTS:
        if Path(p).exists(): return ImageFont.truetype(p,px)
    return ImageFont.load_default()
def render(ch,r):
    im=Image.new('L',(40,56),r.randint(230,255)); d=ImageDraw.Draw(im); f=font(r.randint(34,47)); b=d.textbbox((0,0),ch,font=f)
    x=(40-(b[2]-b[0]))//2-b[0]+r.randint(-3,3); y=(56-(b[3]-b[1]))//2-b[1]+r.randint(-3,3); d.text((x,y),ch,font=f,fill=r.randint(0,25))
    im=im.rotate(r.uniform(-6,6),resample=Image.Resampling.BILINEAR,fillcolor=255)
    if r.random()<.55: im=im.filter(ImageFilter.GaussianBlur(r.uniform(0,.8)))
    a=np.array(im); _,bw=cv2.threshold(a,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU); return bw
def feat(bw):
    bw=cv2.resize(bw,(32,48)); hog=cv2.HOGDescriptor((32,48),(16,16),(8,8),(8,8),9); return hog.compute(bw).reshape(-1)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--per-char',type=int,default=120); ap.add_argument('--out',default='models/ocr_svm.joblib'); a=ap.parse_args(); r=random.Random(20261025); X=[]; y=[]
    for c in CHARS:
        for _ in range(a.per_char): X.append(feat(render(c,r))); y.append(c)
    X=np.asarray(X); y=np.asarray(y); Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,random_state=20261025,stratify=y)
    m=LinearSVC(C=3.0,dual='auto',max_iter=8000); m.fit(Xtr,ytr); pred=m.predict(Xv); acc=accuracy_score(yv,pred); print('validation_accuracy',round(acc,4),'samples',len(y))
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); joblib.dump({'model':m,'chars':CHARS,'validation_accuracy':float(acc),'seed':20261025},a.out); print('saved',a.out)
if __name__=='__main__': main()
