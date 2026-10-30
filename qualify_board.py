# -*- coding: utf-8 -*-
"""Real AX7020 qualification. Stop the application service before running."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
from analysis_core import software_frame
from engine import Engine
from pages import render
from signal_quality import assess_signal_quality
from video_output import configure_video, encode_rgb

STOP = False
def request_stop(*_):
    global STOP
    STOP = True

def save(path,value):
    def encode(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=encode),encoding='utf8')

def distribution(values):
    a=np.asarray(values,dtype=float)
    return {'count':len(a),'p50_ms':float(np.percentile(a,50)),
            'p95_ms':float(np.percentile(a,95)),'p99_ms':float(np.percentile(a,99)),
            'max_ms':float(a.max()),'min_ms':float(a.min()),'mean_ms':float(a.mean())}

def main():
    for name in ('stdout','stderr'):
        stream=getattr(sys,name)
        if getattr(stream,'buffer',None) is not None:
            setattr(sys,name,io.TextIOWrapper(stream.buffer,encoding='utf8',errors='replace',line_buffering=True))
    parser=argparse.ArgumentParser()
    parser.add_argument('--stress-seconds',type=float,default=600)
    parser.add_argument('--out',default='evidence/qualification_20261023')
    args=parser.parse_args()
    os.chdir(str(Path(__file__).resolve().parent))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    from pynq.lib.video.hierarchies import VideoOut
    from vib_driver import VibAccel
    acc=VibAccel('vib_accel.bit','fir_coef_q15.npy')
    engine=Engine(acc,'dataset_test_3ch.npz','svm_model_multich.npz','fir_coef_q15.npy')
    display=acc.ol.hdmi_out
    config=configure_video(display,acc.ol.axi_vdma,1280,720)
    state={'frame_count':len(engine.data),'resources':json.loads(Path('resources.json').read_text()),
           'source':'dataset_test_3ch.npz','busy':True,'progress':(0,0),'playing':False}
    last_paint=[0.]
    def progress(done,total,message=''):
        if time.monotonic()-last_paint[0]>1 or done==total:
            state['progress']=(done,total)
            image=render(5,state,.04)
            frame=display.newframe();np.copyto(frame,encode_rgb(image));display.writeframe(frame)
            last_paint[0]=time.monotonic()
            print(message or 'Progress %d/%d'%(done,total),flush=True)
            save(out/'progress.json',{'stage':message,'done':done,'total':total,'hardware_frames':acc.frames_completed})
    summary={'started_host_label':'2026-10-23','board_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'bitstream_sha256':hashlib.sha256(Path('vib_accel.bit').read_bytes()).hexdigest(),
             'model_sha256':hashlib.sha256(Path('svm_model_multich.npz').read_bytes()).hexdigest(),
             'dataset_sha256':hashlib.sha256(Path('dataset_test_3ch.npz').read_bytes()).hexdigest(),
             'video':config,'test_scope':'Stored-file replay, no ADC, not an early-degradation validation'}
    try:
        print('STAGE full-set accuracy',flush=True)
        accuracy=engine.accuracy_job(progress=lambda d,t:progress(d,t,'Full set %d/%d'%(d,t)))
        save(out/'accuracy.json',accuracy)
        with (out/'accuracy_groups.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(accuracy['records'][0]));w.writeheader();w.writerows(accuracy['records'])
        summary['accuracy']={k:v for k,v in accuracy.items() if k not in ('records',)}
        assert accuracy['groups']==len(engine.groups)
        assert accuracy['pl_ps_match']==accuracy['groups'] and accuracy['rejected']==0
        print('PL/PS match; measured accuracy %d/%d'%(accuracy['correct'],accuracy['groups']),flush=True)

        print('STAGE signal guard and robustness',flush=True)
        rng=np.random.RandomState(20261023)
        quality_rows=[]
        base=engine.data[engine.groups[0][0]]
        invalids={'all_zero':np.zeros_like(base),'flat_dc':np.full_like(base,12000),
                  'rail_clip':np.full_like(base,32767),'one_channel_dropout':base.copy(),
                  'very_low_amplitude':np.round(base.astype(float)*.0001).astype(np.int16)}
        invalids['one_channel_dropout'][:,1]=0
        for name,frames in invalids.items():
            before=acc.frames_completed
            result=engine.diagnose_data(frames,True)
            quality_rows.append({'condition':name,'accepted':result['accepted'],
                                 'prediction':result['prediction'],'reasons':result['quality']['reasons'],
                                 'hardware_frames_used':acc.frames_completed-before})
            assert not result['accepted'] and result['prediction'] is None and acc.frames_completed==before
        save(out/'quality_guards.json',quality_rows)
        robust_rows=[]
        # Fixed predefined perturbations; do not tune the guard to the held-out labels.
        choices=[(ids,truth) for ids,truth in engine.groups]
        selected=[]
        for cls in range(5):
            cls_choices=[(ids,truth) for ids,truth in choices if truth==cls]
            positions=np.linspace(0,len(cls_choices)-1,5,dtype=int)
            selected.extend(cls_choices[int(i)] for i in positions)
        for condition in ('clean','snr20','snr10','snr0','gain2'):
            for ids,truth in selected:
                original=engine.data[ids]
                values=original.astype(float)
                if condition.startswith('snr'):
                    snr=float(condition[3:])
                    rms=np.sqrt(np.mean(values**2,axis=(0,2),keepdims=True))
                    values=values+rng.normal(size=values.shape)*rms/(10**(snr/20.))
                elif condition=='gain2': values*=2.
                perturbed=np.clip(np.round(values),-32768,32767).astype(np.int16)
                result=engine.diagnose_data(perturbed,True)
                robust_rows.append({'condition':condition,'truth':int(truth),
                                    'source':str(engine.source_ids[ids[0]]),'first_index':int(ids[0]),
                                    'prediction':result['prediction'],'accepted':result['accepted'],
                                    'correct':result['prediction']==truth,'elapsed_ms':result['elapsed_ms']})
            print('Robustness '+condition,flush=True)
        save(out/'robustness_groups.json',robust_rows)
        robust_summary={}
        for condition in sorted(set(row['condition'] for row in robust_rows)):
            rows=[r for r in robust_rows if r['condition']==condition]
            robust_summary[condition]={'groups':len(rows),'correct':sum(r['correct'] for r in rows),
                                       'rejected':sum(not r['accepted'] for r in rows),
                                       'accuracy_with_rejections_as_errors':sum(r['correct'] for r in rows)/float(len(rows))}
        summary['robustness']=robust_summary
        save(out/'robustness_summary.json',robust_summary)
        summary['quality_guards_passed']=len(quality_rows)

        print('STAGE same-task benchmark, raw repeated samples',flush=True)
        timings=[]
        frame_ids=np.linspace(0,len(engine.data)-1,40,dtype=int)
        acc.run_frame(engine.data[0,0]);software_frame(engine.data[0,0],engine.dg)
        for round_index in range(3):
            for route in ('pl','ps') if round_index%2==0 else ('ps','pl'):
                for index in frame_ids:
                    frame=engine.data[int(index),0]
                    start=time.perf_counter()
                    if route=='pl':acc.run_frame(frame)
                    else:software_frame(frame,engine.dg)
                    timings.append({'task':'frame_features','route':route,'round':round_index,
                                    'sample_index':int(index),'elapsed_ms':(time.perf_counter()-start)*1000})
                for number in np.linspace(0,len(engine.groups)-1,10,dtype=int):
                    ids,truth=engine.groups[int(number)]
                    result=engine.diagnose_data(engine.data[ids],route=='pl')
                    timings.append({'task':'group_diagnosis','route':route,'round':round_index,
                                    'sample_index':int(ids[0]),'elapsed_ms':result['elapsed_ms']})
        with (out/'latency_samples.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(timings[0]));w.writeheader();w.writerows(timings)
        benchmark={}
        for task in ('frame_features','group_diagnosis'):
            benchmark[task]={route:distribution([v['elapsed_ms'] for v in timings if v['task']==task and v['route']==route])
                             for route in ('pl','ps')}
            benchmark[task]['speedup_p50']=benchmark[task]['ps']['p50_ms']/benchmark[task]['pl']['p50_ms']
        summary['benchmark']=benchmark
        save(out/'benchmark.json',benchmark)

        print('STAGE golden feature comparison',flush=True)
        golden=[]
        for index in np.linspace(0,len(engine.data)-1,30,dtype=int):
            frame=engine.data[int(index),0]
            actual=acc.run_frame(frame);reference=software_frame(frame,engine.dg)
            row={'index':int(index),'rms2_error':abs(actual['rms2']-reference['rms2']),
                 'band_l1_error':float(np.sum(np.abs(np.asarray(actual['band_ratio'])-reference['band_ratio']))),
                 'centroid_error_hz':abs(actual['centroid_hz']-reference['centroid_hz']),
                 'peak_error_hz':abs(actual['peak_hz']-reference['peak_hz']),
                 'kurtosis_error':abs(actual['kurtosis']-reference['kurtosis'])}
            if row['peak_error_hz']>12000/1024:
                for name,feats in [('pl',actual),('ps',reference)]:
                    spec=np.asarray(feats['spectrum']);top=np.argsort(spec[1:])[-5:]+1
                    row[name+'_top_bins']=top.tolist()
                    row[name+'_top_power']=spec[top].tolist()
            golden.append(row)
        assert max(r['rms2_error'] for r in golden)<1e-12
        save(out/'golden_feature_errors.json',golden)
        filt=engine.filter_job(0);save(out/'fir_filter.json',filt)
        assert filt['attenuation_db']>25
        summary['golden_max_errors']={k:max(r[k] for r in golden) for k in
            ('rms2_error','band_l1_error','centroid_error_hz','peak_error_hz','kurtosis_error')}
        summary['fir_suppression_db']=filt['attenuation_db']

        print('STAGE paced replay %.1f seconds'%args.stress_seconds,flush=True)
        period=8*engine.metadata['hop_samples']/12000.
        replay=[]
        epoch=time.monotonic()
        number=0
        with (out/'paced_replay.csv').open('w',newline='') as f:
            fieldnames=['group','truth','prediction','source','release_s','start_s','finish_s',
                        'compute_ms','response_ms','start_lateness_ms','deadline_missed','queue_backlog_groups']
            w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader()
            while not STOP and number*period<args.stress_seconds:
                release=epoch+number*period
                remaining=release-time.monotonic()
                if remaining>0:time.sleep(remaining)
                ids,truth=engine.groups[number%len(engine.groups)]
                start=time.monotonic()
                result=engine.diagnose_data(engine.data[ids],True)
                finish=time.monotonic()
                row={'group':number,'truth':int(truth),'prediction':result['prediction'],
                     'source':str(engine.source_ids[ids[0]]),'release_s':number*period,
                     'start_s':start-epoch,'finish_s':finish-epoch,
                     'compute_ms':(finish-start)*1000,'response_ms':(finish-release)*1000,
                     'start_lateness_ms':max(0,start-release)*1000,
                     'deadline_missed':finish>release+period,
                     'queue_backlog_groups':int(max(0,start-release)//period)}
                replay.append(row);w.writerow(row)
                if number%30==0:
                    f.flush()
                    progress(int(min(args.stress_seconds,finish-epoch)),int(args.stress_seconds),
                             'Paced replay %.0f/%.0f seconds'%(finish-epoch,args.stress_seconds))
                number+=1
        completed=time.monotonic()-epoch
        stress={'requested_duration_s':args.stress_seconds,'observed_duration_s':completed,
                'period_ms':period*1000,'groups':len(replay),'software_skipped_groups':0,
                'deadline_misses':sum(r['deadline_missed'] for r in replay),
                'max_queue_backlog_groups':max(r['queue_backlog_groups'] for r in replay),
                'correct':sum(r['truth']==r['prediction'] for r in replay),
                'response':distribution([r['response_ms'] for r in replay]),
                'compute':distribution([r['compute_ms'] for r in replay]),
                'scope':'Single-worker paced DDR file replay, includes scheduled start lateness; no physical ADC producer',
                'completed':not STOP and number*period>=args.stress_seconds,
                'simulated_first_group_collection_ms':(1024+7*512)/12000*1000}
        # Keep the observation window at least the requested duration, including
        # the tail after the last scheduled release has completed.
        if not STOP and completed < args.stress_seconds:
            time.sleep(args.stress_seconds-completed)
            stress['observed_duration_s']=time.monotonic()-epoch
        summary['stress']=stress;summary['hardware_frames_completed']=acc.frames_completed
        summary['passed']=stress['completed'] and stress['deadline_misses']==0
        save(out/'paced_replay_summary.json',stress)
        save(out/'summary.json',summary)
        state.update(busy=False,accuracy=accuracy)
        image=render(5,state,.04)
        image.save(str(out/'evaluation_framebuffer.png'))
        print('QUALIFICATION COMPLETE '+json.dumps({'passed':summary['passed'],'groups':len(replay),
               'deadline_misses':stress['deadline_misses'],'correct':accuracy['correct']}),flush=True)
    except Exception as exc:
        summary.update(passed=False,error=repr(exc),traceback=traceback.format_exc())
        save(out/'summary.json',summary)
        raise
    finally:
        display.close();acc.close()

if __name__=='__main__':
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    main()

