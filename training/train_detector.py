"""Optional neural detector training. Requires requirements-train.txt."""
import argparse

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='yolo_dataset/data.yaml'); ap.add_argument('--epochs',type=int,default=80); ap.add_argument('--imgsz',type=int,default=640); ap.add_argument('--model',default='yolo11n.pt'); a=ap.parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('Install training dependencies: pip install -r requirements-train.txt')
    model=YOLO(a.model); model.train(data=a.data,epochs=a.epochs,imgsz=a.imgsz,batch=8,device=0,patience=15,workers=2)
if __name__=='__main__': main()
