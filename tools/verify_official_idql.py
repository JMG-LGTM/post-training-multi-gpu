#!/usr/bin/env python3
"""Run official-IDQL numerical checks on two virtual CPU devices only.

This verifies source identity and numerical execution, not CUDA readiness,
robot rollout performance, or the equivalence of unspecified image features.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source_files():
 files=list((ROOT/'src/post_training/official_idql').glob('*.py'))
 files.extend([ROOT/'src/post_training/_vendor/idql_official/UPSTREAM.json',
  ROOT/'tests/test_official_idql_core.py',Path(__file__).resolve()])
 for entry in [ROOT/'train_idql_official.py',ROOT/'scripts/plot_official_idql_metrics.py']:
  if entry.exists():files.append(entry)
 files.extend((ROOT/'configs/official_idql').glob('*.yaml'))
 return {str(path.relative_to(ROOT)):digest(path) for path in sorted(files)}


def parse_affinity(value):
 result=set()
 for item in value.split(','):
  if '-' in item:
   first,last=(int(x) for x in item.split('-',1))
   if first>last:raise ValueError('CPU affinity range is reversed')
   result.update(range(first,last+1))
  else:result.add(int(item))
 if not result or min(result)<0:raise ValueError('CPU affinity must contain nonnegative CPU indices')
 return result

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=ROOT/'reports/official_idql_cpu_verification.json')
 p.add_argument('--cpu-affinity',help='Optional existing CPU subset, e.g.0-7 or32,34; default respects the current cpuset')
 args=p.parse_args(argv);output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
 if hasattr(os,'sched_getaffinity'):
  allowed=os.sched_getaffinity(0)
  requested=parse_affinity(args.cpu_affinity) if args.cpu_affinity else allowed
  if not requested<=allowed:raise RuntimeError('Requested CPU affinity is outside this process allowed cpuset')
  affinity=sorted(requested)
  if args.cpu_affinity:os.sched_setaffinity(0,affinity)
 else:affinity=None
 # Environment is set before the child can import JAX or any model module.
 env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',CUDA_VISIBLE_DEVICES='',
  JAX_ENABLE_X64='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 flags=env.get('XLA_FLAGS','').split()
 flags=[v for v in flags if not v.startswith('--xla_force_host_platform_device_count=')]
 env['XLA_FLAGS']=' '.join(flags+['--xla_force_host_platform_device_count=2'])
 env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
 with tempfile.TemporaryDirectory(prefix='official-idql-cpu-') as directory:
  sources_before=source_files()
  temp=Path(directory);metrics=temp/'measurements.json';junit=temp/'junit.xml'
  env['OFFICIAL_IDQL_TEST_METRICS_PATH']=str(metrics)
  command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(ROOT/'tests/test_official_idql_core.py'),
   '--junitxml='+str(junit)]
  result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
  sources_after=source_files()
  measured=json.loads(metrics.read_text()) if metrics.exists() else {}
  suites=ET.parse(junit).getroot() if junit.exists() else None
  counts={name:sum(int(s.get(name,0)) for s in suites.iter('testsuite')) if suites is not None else None
   for name in ['tests','errors','failures','skipped']}
  passed=result.returncode==0 and bool(measured) and bool(counts['tests']) and counts['errors']==counts['failures']==0
  # Fail closed when execution silently skipped any required numerical test.
  passed=passed and counts['skipped']==0 and counts['tests']==8 and sources_before==sources_after
  report={'schema_version':1,'status':'PASS_OFFICIAL_IDQL_SOURCE_AND_NUMERICS_CPU_ONLY' if passed else 'FAIL_OFFICIAL_IDQL_CPU_VERIFICATION',
   'observed_at':datetime.now(timezone.utc).isoformat(),'python_executable':sys.executable,'python_version':sys.version,
   'source_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
   'test_counts':counts,'command':command,'returncode':result.returncode,'measurements':measured,
   'cpu_affinity':affinity,
   'source_files_sha256':sources_after,'source_files_unchanged_during_verification':sources_before==sources_after,
   'test_source_sha256':digest(ROOT/'tests/test_official_idql_core.py'),'verification_tool_sha256':digest(Path(__file__)),
   'environment':{k:env[k] for k in ['JAX_PLATFORMS','JAX_PLATFORM_NAME','CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_ENABLE_X64']},
   'stdout':result.stdout,'stderr':result.stderr,'gpu_initialized':False,'gpu_numerical_equivalence_claimed':False,
   'training_or_evaluation_started':False,'limitations':['CPU vectors do not validate any unconfigured visual feature extractor.',
    'Virtual CPU pmap does not establish CUDA performance or GPU numerical trajectories.',
    'Reference construction uses Gym Box spaces only; this environment is not a reproduction of the upstream MuJoCo dependency stack.']}
  temporary=output.with_name(output.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');os.replace(temporary,output)
  print(json.dumps({'status':report['status'],'receipt':str(output),'receipt_sha256':digest(output),'test_counts':counts,'returncode':result.returncode}),flush=True)
  if result.returncode:print(result.stdout+result.stderr,file=sys.stderr)
  return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
