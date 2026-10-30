"""Download manifest files from the original CWRU host and verify SHA256."""
import argparse,hashlib,json
from pathlib import Path
from urllib.request import urlopen

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
records=json.loads((Path(__file__).parents[1]/'data/raw_manifest.json').read_text(encoding='utf8'))
for record in records:
    target=a.out/record['file']
    if target.exists():data=target.read_bytes()
    else:
        with urlopen('https://engineering.case.edu/sites/default/files/'+record['file'],timeout=60) as response:
            data=response.read()
    if hashlib.sha256(data).hexdigest()!=record['sha256']:
        raise ValueError('Source changed or corrupt: '+record['file'])
    if not target.exists():target.write_bytes(data)
    print('VERIFIED '+record['file'],flush=True)
