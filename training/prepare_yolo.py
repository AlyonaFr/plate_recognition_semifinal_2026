"""Convert meta.csv bbox annotations to YOLO detection format.
Classes: 0=type1,1=type1a,2=type1b,3=other.
"""
import argparse,csv,shutil
from pathlib import Path
from PIL import Image
MAP={'type1':0,'type1a':1,'type1b':2,'other':3}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='dataset'); ap.add_argument('--out',default='yolo_dataset'); a=ap.parse_args()
    root=Path(a.dataset); out=Path(a.out); (out/'images/train').mkdir(parents=True,exist_ok=True); (out/'labels/train').mkdir(parents=True,exist_ok=True)
    grouped={}
    with (root/'meta.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f,delimiter=';'): grouped.setdefault(r['image'],[]).append(r)
    for rel,rows in grouped.items():
        src=root/rel
        with Image.open(src) as im: W,H=im.size
        dst=out/'images/train'/src.name; shutil.copy2(src,dst)
        lines=[]
        for r in rows:
            x,y,w,h=map(float,r['bbox'].split(',')); xc=(x+w/2)/W; yc=(y+h/2)/H
            lines.append(f"{MAP[r['plate_type']]} {xc:.6f} {yc:.6f} {w/W:.6f} {h/H:.6f}")
        (out/'labels/train'/(src.stem+'.txt')).write_text('\n'.join(lines),encoding='utf-8')
    (out/'data.yaml').write_text("path: .\ntrain: images/train\nval: images/train\nnames: [type1, type1a, type1b, other]\n",encoding='utf-8')
    print(out/'data.yaml')
if __name__=='__main__': main()
