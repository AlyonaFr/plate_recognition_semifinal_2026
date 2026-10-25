import argparse, csv, time
from pathlib import Path
from app.pipeline import PlatePipeline
from app.utils import imread_unicode


def parse_args():
    p=argparse.ArgumentParser(description='Offline recognition of Russian type1/type1a/type1b plates')
    p.add_argument('--input', required=True, help='Directory with .jpg/.jpeg/.png images')
    p.add_argument('--output', required=True, help='Output CSV path')
    p.add_argument('--benchmark', action='store_true', help='Print average processing time')
    return p.parse_args()


def run(input_dir, output_csv, benchmark=False):
    input_dir=Path(input_dir); output_csv=Path(output_csv)
    if not input_dir.is_dir(): raise SystemExit(f'Input directory not found: {input_dir}')
    files=sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}])
    pipe=PlatePipeline(); rows=[]; times=[]
    for p in files:
        img=imread_unicode(p)
        t=time.perf_counter(); result=pipe.process(img); times.append(time.perf_counter()-t)
        for r in result:
            rows.append([p.name,r['plate_num'],r['plate_type'],f"{r['confidence']:.4f}"])
    output_csv.parent.mkdir(parents=True,exist_ok=True)
    with output_csv.open('w',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,delimiter=';'); wr.writerow(['image','plate_num','plate_type','confidence']); wr.writerows(rows)
    if benchmark:
        avg=(sum(times)/len(times)*1000) if times else 0
        print(f'Images: {len(files)}, detections: {len(rows)}, avg: {avg:.2f} ms/image')
    print(f'Saved: {output_csv}')

if __name__=='__main__':
    a=parse_args(); run(a.input,a.output,a.benchmark)
