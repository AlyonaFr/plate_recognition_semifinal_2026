import argparse,csv,re
from pathlib import Path
from collections import Counter
VALID_TYPES={'type1','type1a','type1b','other'}
LETTERS='ABEKMHOPCTYX'
PLATE_RE=re.compile(rf'^[{LETTERS}][0-9]{{3}}[{LETTERS}]{{2}}(?:[0-9]{{2}}|[127][0-9]{{2}})$')
PLATE_PARTIAL_RE=re.compile(rf'^[{LETTERS}#][0-9#]{{3}}[{LETTERS}#]{{2}}(?:[0-9#]{{2}}|[127#][0-9#]{{2}})$')
FIELDS=['image','plate_num','plate_type','bbox','quad','is_vehicle','is_synthetic','source','license','conditions']
RECOMMENDED_REAL={'type1a':150,'type1b':300,'other':50}

def ints(s,n):
    try:
        v=[int(float(x)) for x in s.split(',')]
        return v if len(v)==n else None
    except Exception:return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='dataset'); ap.add_argument('--strict-real-minimums',action='store_true'); a=ap.parse_args(); root=Path(a.root); meta=root/'meta.csv'
    errors=[];warnings=[]; counts=Counter(); real=Counter(); synthetic=Counter(); seen=set()
    if not meta.exists(): raise SystemExit('meta.csv not found')
    with meta.open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f,delimiter=';'); missing=[x for x in FIELDS if x not in (rd.fieldnames or [])]
        if missing: errors.append('Missing fields: '+','.join(missing))
        for i,row in enumerate(rd,2):
            rel=row.get('image',''); p=root/rel; typ=row.get('plate_type',''); counts[typ]+=1
            if row.get('is_synthetic')=='1': synthetic[typ]+=1
            else: real[typ]+=1
            key=(rel,row.get('plate_num',''),row.get('bbox',''))
            if key in seen: errors.append(f'line {i}: duplicate annotation {key}')
            seen.add(key)
            if not rel or rel.startswith('/') or '..' in Path(rel).parts: errors.append(f'line {i}: image path must be relative and safe: {rel}')
            if not p.exists(): errors.append(f'line {i}: image missing: {p}')
            if typ not in VALID_TYPES: errors.append(f'line {i}: bad type {typ}')
            num=row.get('plate_num','').strip()
            if typ!='other':
                if '#' in num:
                    if not PLATE_PARTIAL_RE.fullmatch(num): errors.append(f'line {i}: bad partial plate_num {num}')
                elif not PLATE_RE.fullmatch(num): errors.append(f'line {i}: bad plate_num {num}')
            b=ints(row.get('bbox',''),4); q=ints(row.get('quad',''),8)
            if not b or b[2]<=0 or b[3]<=0: errors.append(f'line {i}: bad bbox {row.get("bbox","")}')
            if not q: errors.append(f'line {i}: bad quad {row.get("quad","")}')
            if p.exists() and b:
                try:
                    import cv2
                    im=cv2.imread(str(p));
                    if im is None: errors.append(f'line {i}: unreadable image {p}')
                    else:
                        h,w=im.shape[:2]; x,y,bw,bh=b
                        if x<0 or y<0 or x+bw>w or y+bh>h: errors.append(f'line {i}: bbox outside image ({w}x{h}): {b}')
                        if q and any(q[j]<0 or q[j]>(w if j%2==0 else h) for j in range(8)): errors.append(f'line {i}: quad outside image ({w}x{h}): {q}')
                except Exception as e: warnings.append(f'line {i}: image-bound check skipped: {e}')
            for flag in ('is_vehicle','is_synthetic'):
                if row.get(flag) not in ('0','1'): errors.append(f'line {i}: {flag} must be 0/1')
            if not row.get('source','').strip(): errors.append(f'line {i}: empty source')
            if not row.get('license','').strip(): errors.append(f'line {i}: empty license')
    if sum(synthetic.values())<5000: warnings.append(f'synthetic images {sum(synthetic.values())} < recommended 5000')
    for typ,n in RECOMMENDED_REAL.items():
        if real[typ]<n:
            msg=f'real {typ}: {real[typ]} < recommended {n}'
            (errors if a.strict_real_minimums else warnings).append(msg)
    print('Counts:',dict(counts)); print('Real:',dict(real)); print('Synthetic:',dict(synthetic)); print('Errors:',len(errors)); print('Warnings:',len(warnings))
    for e in errors[:100]: print('ERROR -',e)
    for w in warnings[:100]: print('WARN  -',w)
    raise SystemExit(1 if errors else 0)
if __name__=='__main__':main()
