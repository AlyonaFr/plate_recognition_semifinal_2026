import argparse,csv,random
from pathlib import Path
import cv2,numpy as np,joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_recall_fscore_support

def feat(img):
    if img is None or img.size==0: return None
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    g=cv2.resize(g,(96,48)); g=cv2.equalizeHist(g)
    hog=cv2.HOGDescriptor((96,48),(16,16),(8,8),(8,8),9)
    f=hog.compute(g).reshape(-1)
    if img.ndim==3:
        hsv=cv2.cvtColor(cv2.resize(img,(96,48)),cv2.COLOR_BGR2HSV)
        color=np.array([hsv[:,:,1].mean()/255,hsv[:,:,2].mean()/255,((hsv[:,:,0]>=14)&(hsv[:,:,0]<=42)&(hsv[:,:,1]>65)).mean()],dtype=np.float32)
        f=np.concatenate([f,color])
    else: f=np.concatenate([f,np.zeros(3,dtype=np.float32)])
    return f

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b; x1=max(ax,bx);y1=max(ay,by);x2=min(ax+aw,bx+bw);y2=min(ay+ah,by+bh); inter=max(0,x2-x1)*max(0,y2-y1); return inter/(aw*ah+bw*bh-inter+1e-6)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='dataset'); ap.add_argument('--limit',type=int,default=1600); ap.add_argument('--out',default='models/detector_svm.joblib'); a=ap.parse_args(); root=Path(a.root); r=random.Random(20261025); rows=[]
    with (root/'meta.csv').open(encoding='utf-8-sig') as f:
        rd=csv.DictReader(f,delimiter=';'); rows=[x for x in rd if x['plate_type']!='other'][:a.limit]
    X=[];Y=[]
    for row in rows:
        im=cv2.imread(str(root/row['image']));
        if im is None: continue
        x,y,w,h=map(int,row['bbox'].split(',')); H,W=im.shape[:2]
        pad=max(2,int(.05*max(w,h))); xx=max(0,x-pad);yy=max(0,y-pad);x2=min(W,x+w+pad);y2=min(H,y+h+pad)
        q=feat(im[yy:y2,xx:x2]);
        if q is not None: X.append(q);Y.append(1)
        for _ in range(2):
            for __ in range(20):
                nw=r.randint(max(30,w//2),min(W,max(40,w*2))); nh=r.randint(max(18,h//2),min(H,max(24,h*2))); nx=r.randint(0,max(0,W-nw)); ny=r.randint(0,max(0,H-nh))
                if iou((nx,ny,nw,nh),(x,y,w,h))<.08: break
            q=feat(im[ny:ny+nh,nx:nx+nw]);
            if q is not None: X.append(q);Y.append(0)
    X=np.asarray(X);Y=np.asarray(Y); Xtr,Xv,ytr,yv=train_test_split(X,Y,test_size=.2,random_state=20261025,stratify=Y)
    m=LinearSVC(C=1.2,dual='auto',class_weight='balanced',max_iter=6000);m.fit(Xtr,ytr);p=m.predict(Xv);acc=accuracy_score(yv,p);pr,rc,f1,_=precision_recall_fscore_support(yv,p,average='binary',zero_division=0)
    print('val',{'accuracy':round(acc,4),'precision':round(pr,4),'recall':round(rc,4),'f1':round(f1,4),'samples':len(Y)})
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);joblib.dump({'model':m,'metrics':{'accuracy':float(acc),'precision':float(pr),'recall':float(rc),'f1':float(f1)},'seed':20261025,'feature':'HOG+HSV','training':'synthetic plate crops + random negatives'},a.out);print('saved',a.out)
if __name__=='__main__':main()
