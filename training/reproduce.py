"""Rebuild recordings, grouped features and NumPy SVM exports from CWRU MAT files.

Default fixes the normal-baseline 48 kHz -> 12 kHz rate mismatch. The legacy
profile exists only to reproduce the earlier artifact, never for physical Hz claims.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis_core import software_frame, _groups
from diagnosis import Diagnoser

NORMAL = {97: 0, 98: 1, 99: 2, 100: 3}
FAULT = {}
for ids, cls in [([105,106,107,108],1),([169,170,171,172],1),
                 ([118,119,120,121],2),([185,186,187,188],2),
                 ([130,131,132,133],3),([197,198,199,200],3)]:
    FAULT.update({fid: (cls, load) for load, fid in enumerate(ids)})


def synth_unbalance(n, rng):
    frames = []
    for _ in range(n):
        frot, amp = rng.uniform(28.,30.), rng.uniform(.5,1.5)
        t = np.arange(1024) / 12000.
        channels = []
        for gain, phase in ((1.,0.),(.8,.4),(.6,.8)):
            x = (amp*gain*np.sin(2*np.pi*frot*t+phase)
                 + .30*amp*gain*np.sin(2*np.pi*2*frot*t+2*phase)
                 + .08*amp*gain*np.sin(2*np.pi*3*frot*t+3*phase)
                 + rng.normal(0,.15*amp,1024))
            channels.append((x*(1+.05*np.sin(2*np.pi*2*t))).astype(np.float32))
        frames.append(np.stack(channels))
    return np.stack(frames)


def read_frames(raw, fid, profile):
    path = raw / ('%d.mat'%fid)
    m = loadmat(str(path))
    keys = {c: 'X%03d_%s_time'%(fid,c) for c in ('DE','FE','BA')}
    selected = {c:k for c,k in keys.items() if k in m}
    if not all(c in selected for c in ('DE','FE')):
        raise ValueError('Missing exact file-ID DE/FE keys: '+str(path))
    channels = {c:m[k].ravel().astype(np.float32) for c,k in selected.items()}
    native_fs = 48000 if fid in NORMAL else 12000
    if profile == 'resampled' and native_fs != 12000:
        channels = {c:resample_poly(x,1,4).astype(np.float32) for c,x in channels.items()}
    if 'BA' not in channels:
        channels['BA'] = np.roll(channels['DE'],512)
    n = min(map(len,channels.values()))
    indices = np.arange(1024)[None,:] + 512*np.arange((n-1024)//512+1)[:,None]
    frames = np.stack([channels[c][:n][indices] for c in ('DE','FE','BA')],axis=1)
    record = {'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'selected_variables':selected,'original_lengths':{c:int(m[k].size) for c,k in selected.items()},
              'native_fs_hz':native_fs,'effective_fs_hz':12000 if profile=='resampled' else native_fs,
              'processing':'resample_poly(up=1,down=4,default Kaiser window)' if native_fs==48000 and profile=='resampled' else 'none',
              'derived_ba':'roll(DE,512) after rate conversion' if 'BA' not in selected else None,
              'frames':len(frames)}
    return frames,record


def quantize(x, scale):
    return (np.clip(x/scale,-1.,32767/32768.)*32768).astype(np.int16)


def export(path,scaler,clf):
    np.savez_compressed(str(path),mean=scaler.mean_,scale=scaler.scale_,
        sv=clf.support_vectors_,dual_coef=clf.dual_coef_,intercept=clf.intercept_,
        n_support=clf.n_support_,classes=clf.classes_,gamma=np.float64(clf._gamma))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--profile',choices=['resampled','legacy'],default='resampled')
    args=p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Choose a new empty output folder')
    args.out.mkdir(parents=True,exist_ok=True)
    sources={c:[] for c in range(5)}; records=[]
    for fid in list(NORMAL)+sorted(FAULT):
        frames,record=read_frames(args.raw_dir,fid,args.profile)
        cls,load=(0,NORMAL[fid]) if fid in NORMAL else FAULT[fid]
        record.update(label=cls,load_hp=load,split='test' if load==3 else 'train')
        records.append(record);sources[cls].append((fid,load,frames))
    target=min(sum(len(f) for _,_,f in sources[c]) for c in (1,2,3))
    synthetic=synth_unbalance(target,np.random.default_rng(20261019))
    for load in range(4):
        sources[4].append((4000+load,load,synthetic[np.arange(target)%4==load]))
    scale=max(float(np.max(np.abs(f))) for files in sources.values() for _,load,f in files if load!=3)
    helper=Diagnoser(str(ROOT/'svm_model_multich.npz'),str(ROOT/'fir_coef_q15.npy'))
    vectors={'train':[],'test':[]}; labels={'train':[],'test':[]}; group_sources={'train':[],'test':[]}
    test_x=[];test_y=[];test_origin=[]
    for cls,files in sources.items():
        for fid,load,frames in files:
            split='test' if load==3 else 'train'
            x=quantize(frames,scale)
            if split=='test':
                test_x.append(x);test_y.extend([cls]*len(x));test_origin.extend([fid]*len(x))
            for start in range(0,len(x)-7,8):
                group=x[start:start+8]
                feats=[[software_frame(f,helper) for f in group[:,c]] for c in range(3)]
                raws=[group[:,c].astype(float)/32768. for c in range(3)]
                vectors[split].append(helper.feature_vector_multichannel(feats,raws))
                labels[split].append(cls);group_sources[split].append(fid)
        print('Features complete class %d'%cls,flush=True)
    train,test=np.asarray(vectors['train']),np.asarray(vectors['test'])
    yt,yv=np.asarray(labels['train']),np.asarray(labels['test'])
    report={'profile':args.profile,'q115_scale':scale,'train_groups':len(train),'test_groups':len(test),
            'seed':20261019,'group_size':8,'frame':1024,'hop':512,'fs_hz':12000,
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit_learn':sklearn.__version__},
            'sampling_reference':'Smith and Randall 2015 DOI 10.1016/j.ymssp.2015.04.021 Appendix A Tables A1/A2',
            'notes':['Normal BA derived; unbalance synthetic.','Load 3 held out by physical file; overlapping windows within files.','Held-out test used for evaluation only; no tuning on noise test.']}
    for name,width in [('three_channel',39),('two_channel',26)]:
        scaler=StandardScaler().fit(train[:,:width])
        clf=SVC(C=10,gamma='scale').fit(scaler.transform(train[:,:width]),yt)
        path=args.out/('svm_model_multich.npz' if width==39 else 'two_channel_model.npz')
        export(path,scaler,clf)
        pred=clf.predict(scaler.transform(test[:,:width]))
        portable=Diagnoser(str(path),str(ROOT/'fir_coef_q15.npy')).predict(test[:,:width])[0]
        assert np.array_equal(portable,pred),'NumPy export changed SVM predictions'
        matrix=np.zeros((5,5),dtype=int)
        for truth,guess in zip(yv,pred):matrix[truth,guess]+=1
        report[name]={'correct':int(np.sum(pred==yv)),'groups':len(yv),'accuracy':float(np.mean(pred==yv)),
            'confusion':matrix.tolist(),'support_vectors':int(len(clf.support_vectors_)),'numpy_sklearn_match':True}
    X,y,origin=np.concatenate(test_x),np.asarray(test_y),np.asarray(test_origin)
    dataset_id='cwru_resampled_20261023' if args.profile=='resampled' else 'cwru_legacy_20261019'
    metadata=dict(q115_scale=np.float64(scale),fs_hz=np.float64(12000),hop_samples=np.int64(512),dataset_id=np.asarray(dataset_id))
    np.savez_compressed(str(args.out/'dataset_test_3ch.npz'),X=X,y=y,group_id=origin,**metadata)
    # Contiguous blocks stay within one source. Three 16-frame blocks per class
    # allow independent 8-frame grouping without creating a source discontinuity.
    demo_ids=[]
    for cls in range(5):
        ids=np.where(y==cls)[0]
        for i in (0,len(ids)//3,2*len(ids)//3):
            start=ids[i]
            if len(set(origin[start:start+16]))!=1:
                start=next(j for j in ids[i:] if len(set(origin[j:j+16]))==1 and len(set(y[j:j+16]))==1)
            demo_ids.extend(range(start,start+16))
    di=np.asarray(demo_ids)
    # Separate block IDs prevent joining discontinuous excerpt boundaries.
    np.savez_compressed(str(args.out/'demo_set.npz'),X=X[di],y=y[di],group_id=np.arange(len(di))//16,
                        original_source=origin[di],test_indices=di,**metadata)
    np.savez_compressed(str(args.out/'group_features.npz'),train=train,test=test,y_train=yt,y_test=yv,
                        train_source=group_sources['train'],test_source=group_sources['test'])
    report.update(test_frames=len(X),demo_frames=len(di),demo_groups=len(_groups(y[di],len(di),np.arange(len(di))//16)))
    for name,obj in [('training_report.json',report),('raw_manifest.json',records)]:
        (args.out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
