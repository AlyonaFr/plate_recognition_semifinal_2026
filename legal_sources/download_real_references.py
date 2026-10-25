"""Download curated legal reference images and record license metadata.
Run only during dataset preparation; inference remains completely offline.
The script queries Wikimedia Commons metadata before downloading and refuses
unknown/restrictive licenses. It does NOT silently add files to meta.csv.
"""
import csv,json,re,sys
from pathlib import Path
from urllib.parse import quote,urlencode
from urllib.request import urlopen,Request

FILES={
 'type1a_reference.png':'License plate in Russia 1A.png',
 'type1a_vehicle.jpg':'Moscow, Bestune EO5, May 2025 01.jpg',
 'type1b_reference.png':'Russian license plate (for taxi, buses).png',
 'type1b_photo.jpg':'Licenseplate Russia Buss - Taxi 01.jpg',
 'other_antique.png':'Russian antique vehicle license plate.png',
}
ALLOWED=('cc0','public domain','cc by 4.0','cc by-sa 4.0','cc-by-4.0','cc-by-sa-4.0')
API='https://commons.wikimedia.org/w/api.php'

def get_meta(filename):
    q={'action':'query','format':'json','prop':'imageinfo','iiprop':'url|extmetadata','titles':'File:'+filename}
    req=Request(API+'?'+urlencode(q),headers={'User-Agent':'AISCITY-semifinal-dataset/1.0'})
    data=json.load(urlopen(req,timeout=30)); page=next(iter(data['query']['pages'].values())); ii=page['imageinfo'][0]; em=ii.get('extmetadata',{})
    def v(k): return re.sub('<[^>]+>','',em.get(k,{}).get('value','')).strip()
    return {'filename':filename,'url':ii['url'],'page':'https://commons.wikimedia.org/wiki/File:'+quote(filename.replace(' ','_')),'license':v('LicenseShortName') or v('UsageTerms'),'artist':v('Artist'),'credit':v('Credit'),'description':v('ImageDescription')}

def main():
    out=Path(__file__).resolve().parents[1]/'dataset'/'images'/'real_references';out.mkdir(parents=True,exist_ok=True); rows=[]
    for local,fn in FILES.items():
        m=get_meta(fn); lic=m['license'].lower()
        if not any(x in lic for x in ALLOWED):
            print('SKIP license not allowlisted:',fn,m['license']);continue
        data=urlopen(Request(m['url'],headers={'User-Agent':'AISCITY-semifinal-dataset/1.0'}),timeout=60).read(); (out/local).write_bytes(data); m['local_path']='images/real_references/'+local; rows.append(m); print('OK',local,m['license'])
    with (out/'provenance.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['local_path','filename','page','url','license','artist','credit','description'],delimiter=';');wr.writeheader();wr.writerows(rows)
if __name__=='__main__': main()
