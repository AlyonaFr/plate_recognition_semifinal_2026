"""Collect legally reusable real-camera candidates from Wikimedia Commons.

This is a preparation-time tool. It queries vehicle categories, verifies each
file's license, downloads it, blurs detected faces, uses OUR local detector to
propose plate boxes, and writes challenge-format metadata. Automatic annotation
is explicitly logged in dataset/annotation_audit.csv and is never represented as
manual annotation. Review is still recommended before final submission.
"""
import argparse,csv,json,re,time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen
import cv2,numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.detector import PlateDetector
from app.plate_classifier import PlateTypeClassifier

API='https://commons.wikimedia.org/w/api.php'
UA='AISCITY-semifinal-dataset/1.0 (educational noncommercial dataset)'
ALLOW=('cc0','public domain','cc by 4.0','cc by-sa 4.0','cc-by-4.0','cc-by-sa-4.0')
CATEGORIES={
 'type1a':['Japanese specification automobiles in Russia','Kei cars in Moscow'],
 'type1b':['Buses with license plates of Russia','Taxis in Russia','Marshrutka in Russia'],
}

def api(params):
    p={'format':'json','formatversion':2,**params}; return json.load(urlopen(Request(API+'?'+urlencode(p),headers={'User-Agent':UA}),timeout=45))
def strip_html(s): return re.sub('<[^>]+>','',s or '').strip()
def category_files(cat,limit=500):
    cont=None; out=[]
    while len(out)<limit:
        p={'action':'query','generator':'categorymembers','gcmtitle':'Category:'+cat,'gcmtype':'file','gcmlimit':'50','prop':'imageinfo','iiprop':'url|extmetadata'}
        if cont:p.update(cont)
        d=api(p); out.extend(d.get('query',{}).get('pages',[])); cont=d.get('continue')
        if not cont: break
    return out[:limit]
def meta(page):
    ii=(page.get('imageinfo') or [{}])[0]; em=ii.get('extmetadata',{})
    val=lambda k:strip_html(em.get(k,{}).get('value',''))
    return {'title':page.get('title','').removeprefix('File:'),'url':ii.get('url',''),'page':'https://commons.wikimedia.org/wiki/'+page.get('title','').replace(' ','_'),'license':val('LicenseShortName') or val('UsageTerms'),'artist':val('Artist')}
def allowed(lic): return any(x in lic.lower() for x in ALLOW)
def download(url): return urlopen(Request(url,headers={'User-Agent':UA}),timeout=60).read()
def blur_faces(img):
    try:
        xml=cv2.data.haarcascades+'haarcascade_frontalface_default.xml'; cas=cv2.CascadeClassifier(xml); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        for x,y,w,h in cas.detectMultiScale(gray,1.15,5,minSize=(28,28)):
            roi=img[y:y+h,x:x+w]; img[y:y+h,x:x+w]=cv2.GaussianBlur(roi,(0,0),max(10,min(w,h)/3))
    except Exception: pass
    return img
def hashes_for_type(t): return '########'  # legal partial transcription: unknown positions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='dataset');ap.add_argument('--type1a',type=int,default=150);ap.add_argument('--type1b',type=int,default=300);ap.add_argument('--max-files-per-category',type=int,default=500);a=ap.parse_args()
    root=Path(a.root); out=root/'images'/'real';out.mkdir(parents=True,exist_ok=True); detector=PlateDetector(); classifier=PlateTypeClassifier(); targets={'type1a':a.type1a,'type1b':a.type1b}; got={'type1a':0,'type1b':0}; rows=[];audit=[]; seen=set()
    for wanted,cats in CATEGORIES.items():
        for cat in cats:
            if got[wanted]>=targets[wanted]:break
            try: pages=category_files(cat,a.max_files_per_category)
            except Exception as e: print('category failed',cat,e);continue
            for pg in pages:
                if got[wanted]>=targets[wanted]:break
                m=meta(pg)
                if not m['url'] or not allowed(m['license']) or m['url'] in seen:continue
                seen.add(m['url'])
                try:
                    arr=np.frombuffer(download(m['url']),np.uint8); img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
                    if img is None:continue
                    img=blur_faces(img)
                    dets=detector.detect(img)
                    accepted=[]
                    for d in dets:
                        x,y,w,h=d.bbox; crop=img[y:y+h,x:x+w]; typ,tc=classifier.predict(crop)
                        if typ==wanted and d.score*tc>=0.42: accepted.append((d,tc))
                    if not accepted:continue
                    local=f'{wanted}_commons_{got[wanted]:04d}.jpg'; cv2.imwrite(str(out/local),img,[cv2.IMWRITE_JPEG_QUALITY,92])
                    for d,tc in accepted[:2]:
                        x,y,w,h=d.bbox; q=d.quad.reshape(-1,2); quad=','.join(str(int(round(v))) for pt in q for v in pt)
                        rows.append([f'images/real/{local}',hashes_for_type(wanted),wanted,f'{x},{y},{w},{h}',quad,1,0,m['page'],m['license'],'day'])
                        audit.append([f'images/real/{local}',wanted,'auto_our_detector+rule_classifier',round(d.score*tc,4),m['page'],m['license']])
                    got[wanted]+=1; print(wanted,got[wanted],'/',targets[wanted],m['title'])
                except Exception as e: print('skip',m['title'],e)
                time.sleep(.05)
    if rows:
        with (root/'meta.csv').open('a',encoding='utf-8',newline='') as f: csv.writer(f,delimiter=';').writerows(rows)
    audit_path=root/'annotation_audit.csv'; exists=audit_path.exists()
    with audit_path.open('a',encoding='utf-8',newline='') as f:
        wr=csv.writer(f,delimiter=';');
        if not exists:wr.writerow(['image','proposed_type','annotation_method','proposal_confidence','source','license'])
        wr.writerows(audit)
    print('collected',got,'NOTE: review annotation_audit.csv before submission')
if __name__=='__main__':main()
