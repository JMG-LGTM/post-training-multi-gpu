"""Single-process JAX/pmap official critic runner over the existing native cache.

No OpenPI policy, actor update, preference loss or PyTorch checkpoint is loaded.
Pixel networks are an explicit domain adaptation using upstream network modules.
"""
from __future__ import annotations

import copy
import csv
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import yaml

from ..config import DEFAULTS as REPLAY_DEFAULTS, _merge, validate_config as validate_replay_config
from ..io import atomic_json, sha256_file, stable_hash
from .adapters import ReplayAdapter, file_binding, frozen_membership
from .checkpoint import BACKEND, UPSTREAM_COMMIT, load_checkpoint, save_checkpoint

DEFAULTS = dict(schema_version=1, backend=BACKEND, output_dir='outputs/idql_official', cache_dir='cache',
    adapter=dict(mode=None, vector_source=None, pixels=None, external_features=None),
    data={**copy.deepcopy(REPLAY_DEFAULTS['data']), 'split_strategy':'haizhou_hash_threshold',
          'image_size':128, 'frozen_members':None},
    reward=copy.deepcopy(REPLAY_DEFAULTS['reward']),
    training=dict(optimizer_updates=45000, global_batch_size=64, per_device_batch_size=32,
        gradient_accumulation_steps=1, seed=7, checkpoint_interval=2500, validation_interval=500,
        validation_batch_size=64, log_interval=25, sampling='replacement'),
    critic=dict(hidden_dims=[256,256], critic_lr=3e-4, value_lr=3e-4, expectile=.7, tau=.005),
    runtime=dict(platform='gpu', device_count=2),
    frozen_behavior=dict(policy_id=None, actor_sha256=None))
PIXEL_KEYS = {'architecture','camera_layout','encoder_features','encoder_filters','encoder_strides','latent_dim'}


def require(condition, message):
    if not condition: raise ValueError('Official IDQL runner: ' + message)


def _integer(value, name, minimum=1):
    require(type(value) is int and value >= minimum, name + ' must be an integer >= ' + str(minimum))


def load_config(path):
    value = _merge(DEFAULTS, yaml.safe_load(Path(path).read_text()) or {})
    validate_config(value)
    return value


def replay_config(config):
    """Reuse the exact native cache implementation and fingerprint inputs."""
    value = copy.deepcopy(REPLAY_DEFAULTS)
    value.update(method='critic', output_dir=config['output_dir'], cache_dir=config['cache_dir'])
    value['data'] = {k:v for k,v in config['data'].items() if k not in {'frozen_members','image_size'}}
    value['reward'] = copy.deepcopy(config['reward'])
    value['critic']['image_size'] = config['data']['image_size']
    value['training'].update(global_batch_size=config['training']['global_batch_size'],
        per_device_batch_size=config['training']['per_device_batch_size'], gradient_accumulation_steps=1,
        warmup_updates=0, optimizer_updates=config['training']['optimizer_updates'])
    value['distributed']['device'] = 'cpu'
    return value


def validate_config(config):
    require(config.get('schema_version') == 1 and type(config['schema_version']) is int
            and config.get('backend') == BACKEND, 'unsupported schema/backend')
    for key in ('output_dir','cache_dir'):
        require(isinstance(config[key],str) and bool(config[key].strip()), 'missing '+key)
    runtime, training, adapter = config['runtime'], config['training'], config['adapter']
    require(runtime['platform'] in {'cpu','gpu'}, 'runtime platform must be cpu or gpu')
    _integer(runtime['device_count'],'device_count')
    for key in ('optimizer_updates','global_batch_size','per_device_batch_size','checkpoint_interval',
                'validation_interval','validation_batch_size','log_interval'):
        _integer(training[key],'training.'+key)
    _integer(training['seed'],'training.seed',0)
    require(type(training['gradient_accumulation_steps']) is int and training['gradient_accumulation_steps']==1,
            'this official path supports explicit accumulation1 only')
    require(training['global_batch_size']==training['per_device_batch_size']*runtime['device_count'],
            'global batch must equal per-device batch times JAX device count')
    require(training['sampling']=='replacement','only explicit replacement sampling is supported')
    require(adapter['mode'] in {'vector','external_features','pixels'}, 'adapter.mode must be explicit')
    if adapter['mode']=='vector':
        require(adapter['vector_source']=='normalized_proprio' and adapter['pixels'] is None
                and adapter['external_features'] is None,'vector adapter must explicitly use normalized_proprio only')
    elif adapter['mode']=='pixels':
        pixel=adapter['pixels']
        require(isinstance(pixel,dict) and set(pixel)==PIXEL_KEYS
                and pixel['architecture']=='upstream_pixel_multiplexer_d4pg'
                and pixel['camera_layout']=='separate_rgb_views'
                and adapter['external_features'] is None and adapter['vector_source'] is None,
                'pixels require the explicit upstream separate-RGB network adaptation')
        for key in ('encoder_features','encoder_filters','encoder_strides'):
            require(isinstance(pixel[key],list) and pixel[key] and all(type(x) is int and x>0 for x in pixel[key]),
                    'invalid pixels.'+key)
        require(len(pixel['encoder_features'])==len(pixel['encoder_filters'])==len(pixel['encoder_strides']),
                'pixel convolution specifications differ in length')
        _integer(pixel['latent_dim'],'latent_dim')
    else:
        require(isinstance(adapter['external_features'],dict) and adapter['pixels'] is None
                and adapter['vector_source'] is None,'external features must be explicit')
    critic=config['critic']
    require(isinstance(critic['hidden_dims'],list) and critic['hidden_dims']
            and all(type(x) is int and x>0 for x in critic['hidden_dims']),'invalid critic hidden dimensions')
    for key in ('critic_lr','value_lr','expectile','tau'):
        require(type(critic[key]) in (int,float) and math.isfinite(critic[key]) and critic[key]>0,'invalid critic.'+key)
    require(critic['expectile']<1 and critic['tau']<=1,'invalid expectile/target tau')
    behavior=config['frozen_behavior']
    require((behavior['policy_id'] is None)==(behavior['actor_sha256'] is None),'behavior ID/SHA must be paired')
    if behavior['policy_id'] is not None:
        import re
        require(isinstance(behavior['policy_id'],str) and bool(behavior['policy_id'].strip())
                and re.fullmatch('[0-9a-f]{64}',str(behavior['actor_sha256'])) is not None,'invalid frozen behavior metadata')
    _integer(config['data']['image_size'],'data.image_size')
    require(config['data']['frozen_members'] is None or isinstance(config['data']['frozen_members'],str),
            'frozen_members must be a file path or null')
    validate_replay_config(replay_config(config))


def reject_torchrun():
    require(int(os.environ.get('WORLD_SIZE','1'))==1 and int(os.environ.get('RANK','0'))==0,
            'use one JAX process with pmap; torchrun/WORLD_SIZE>1 is forbidden')


def prepare(config, data_root):
    """Verify native files/cache, normalization and exact full root membership."""
    from ..data import inspect_dataset
    from ..replay import CachedReplayDataset, build_replay_cache
    from .upstream import source_receipt
    validate_config(config); reject_torchrun()
    base=replay_config(config)
    cache,reused=build_replay_cache(data_root,config['cache_dir'],base)
    info=inspect_dataset(data_root,base,verify=False)
    native=CachedReplayDataset(cache,'train'); validation=CachedReplayDataset(cache,'val')
    require(len(native)>0 and len(validation)>0,'native train and held-out sets must be nonempty')
    membership=frozen_membership(config['data']['frozen_members'],info)
    adapter=ReplayAdapter(native,config['adapter'])
    normalization=native.metadata['normalization']
    return dict(cache=str(cache),cache_reused=reused,cache_fingerprint=native.metadata['fingerprint'],
        cache_metadata_sha256=sha256_file(cache/'cache.json'),dataset_sha256=info['dataset_sha256'],
        dataset_manifest=file_binding(Path(data_root)/'manifest.json'),data_root=str(Path(data_root).expanduser().resolve()),
        root_counts=info['root_counts'],train_count=len(native),val_count=len(validation),
        frozen_membership=membership,normalization=normalization,normalization_sha256=stable_hash(normalization),
        normalization_file=file_binding(config['data']['normalization']) if config['data']['normalization'] else None,
        adapter_contract=adapter.contract,upstream=source_receipt(),actor_loaded=False,actor_updates=0)


def run_contract(config,prepared):
    files=list(Path(__file__).parent.glob('*.py'))
    files += [Path(__file__).parent.parent/name for name in ('replay.py','data.py','archive_io.py','config.py','io.py')]
    comparable=copy.deepcopy(config)
    # Exact same-run resume, including source paths, adapter and sampling budget.
    return dict(schema_version=1,backend=BACKEND,upstream_commit=UPSTREAM_COMMIT,
        upstream_manifest_sha256=prepared['upstream']['manifest_sha256'],config=comparable,
        config_sha256=stable_hash(comparable),
        prepared={k:v for k,v in prepared.items() if k not in {'cache_reused','upstream'}},
        source_sha256={str(p.relative_to(Path(__file__).parents[1])):sha256_file(p) for p in files},
        versions={name:importlib.metadata.version(name) for name in ('jax','jaxlib','flax','optax','numpy')},
        actor_loaded=False,actor_updates=0)


def dry_run(config,data_root):
    """CPU-only preparation: no JAX device initialization or model construction."""
    prepared=prepare(config,data_root)
    return dict(status='PASS_OFFICIAL_IDQL_CPU_PREPARATION',prepared=prepared,
        contract=run_contract(config,prepared),backend=BACKEND,actor_loaded=False,actor_updates=0,
        declared_device_count=config['runtime']['device_count'],device_visibility_verified=False,
        gpu_used=False,note='GPU visibility/pmap/throughput require a separate authorized GPU smoke.')


def create_state(config,adapter,devices=None):
    from . import core
    import jax
    sample=adapter.batch(np.array([0],np.int64)); mode=adapter.contract['observation_kind']
    options=dict(config['critic']);options['hidden_dims']=tuple(options['hidden_dims'])
    options.update(discount=config['reward']['discount'],observation_kind=mode,
                   axis_name='devices' if config['runtime']['device_count']>1 else None)
    if mode=='pixels':
        pixel=config['adapter']['pixels'];options.update(pixel_keys=tuple(adapter.contract['pixel_keys']),
            **{key:tuple(pixel[key]) for key in ('encoder_features','encoder_filters','encoder_strides')},latent_dim=pixel['latent_dim'])
    agent=core.create_agent(config['training']['seed'],jax.tree_util.tree_map(lambda x:x[0],sample['observations']),
                            sample['actions'][0],**options)
    return jax.device_put_replicated(agent,devices) if config['runtime']['device_count']>1 else agent


def evaluate(agent,adapter,batch_size):
    from .core import evaluate_losses
    totals={};count=0
    for start in range(0,len(adapter),batch_size):
        indices=np.arange(start,min(start+batch_size,len(adapter)),dtype=np.int64)
        metrics=evaluate_losses(agent,adapter.batch(indices));size=len(indices);count+=size
        for key,value in metrics.items():
            number=float(np.asarray(value));require(math.isfinite(number),'nonfinite held-out '+key)
            totals[key]=totals.get(key,0.)+size*number
    return dict(count=count,held_out=True,updated=False,**{k:v/count for k,v in totals.items()})


def _devices(config):
    reject_torchrun()
    platform=config['runtime']['platform'];requested=config['runtime']['device_count']
    if platform=='cpu':os.environ['JAX_PLATFORMS']='cpu'
    else:
        visible=os.environ.get('CUDA_VISIBLE_DEVICES')
        require(isinstance(visible,str) and bool(visible.strip()),'GPU mode requires explicit CUDA_VISIBLE_DEVICES before JAX initialization')
        ids=[value.strip() for value in visible.split(',')]
        require(len(ids)==requested and all(ids) and '-1' not in ids and len(set(ids))==len(ids),
                'CUDA_VISIBLE_DEVICES count/duplicates differ from declared device_count')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    require(jax.process_count()==1,'multi-process JAX is not supported')
    devices=jax.local_devices(backend=platform)
    require(len(devices)>=requested if platform=='cpu' else len(devices)==requested,
            'visible JAX device count differs; explicitly configure CUDA_VISIBLE_DEVICES for GPU runs')
    return devices[:requested]


def _append_csv(path,row):
    exists=path.exists()
    with path.open('a',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row))
        if not exists:writer.writeheader()
        writer.writerow(row);handle.flush();os.fsync(handle.fileno())


def _trim_history(directory,step):
    """Keep only actual records at/before the restored boundary; never invent loss."""
    path=directory/'metrics.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    rows=[row for row in rows if row['optimizer_update']<=step]
    require([row['optimizer_update'] for row in rows]==list(range(1,step+1)),
            'same-run raw history is missing or duplicated before the checkpoint boundary')
    path.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows))
    for name in ('metrics.csv','validation.csv'):
        p=directory/name
        if p.exists():
            with p.open() as stream:reader=csv.DictReader(stream);fields=reader.fieldnames;values=list(reader)
            with p.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
                writer.writerows(row for row in values if int(row['optimizer_update'])<=step)


def train(config,data_root,*,resume=None,stop_after=None):
    """Complete fresh/same-run restore loop; stop_after is a CPU-test boundary."""
    validate_config(config);reject_torchrun()
    directory=Path(config['output_dir']).expanduser().absolute();directory.parent.mkdir(parents=True,exist_ok=True)
    with (directory.parent/('.'+directory.name+'.official.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if resume is None:require(not directory.exists() or not any(directory.iterdir()),'fresh output directory is not empty')
        else:
            from .checkpoint import _path
            requested=_path(resume)  # Refuse old .pt before model/device initialization.
            require(requested.parent==directory/'checkpoints','resume must be this run\'s official checkpoint')
        prepared=prepare(config,data_root);devices=_devices(config)
        import jax
        from ..replay import CachedReplayDataset
        from .core import update
        native=ReplayAdapter(CachedReplayDataset(prepared['cache'],'train'),config['adapter'])
        heldout=ReplayAdapter(CachedReplayDataset(prepared['cache'],'val'),config['adapter'])
        state=create_state(config,native,devices);contract=run_contract(config,prepared)
        generator=np.random.default_rng(config['training']['seed']+1);start=0;draws=0;metadata={'segments':0}
        if resume is not None:
            state,sampling,metadata=load_checkpoint(resume,state,contract)
            generator.bit_generator.state=sampling['rng_state'];start=int(np.asarray(state.step).flat[0]);draws=sampling['draws']
            require(sampling['completed_updates']==start and draws==start*config['training']['global_batch_size'],
                    'restored sampler/update counters differ')
            require(Path(resume).name==f'update_{start:08d}','checkpoint directory/update identity differs')
            for group in ('checkpoints','scorers'):
                for path in (directory/group).glob('update_*'):
                    suffix=path.name.removeprefix('update_')
                    require(suffix.isdigit() and int(suffix)<=start,
                            'resume must use latest boundary; immutable future artifacts are retained')
            _trim_history(directory,start)
            stop_file=directory/'stop.request'
            if stop_file.exists():
                require(not stop_file.is_symlink() and stop_file.is_file(),'unsafe stop request')
                metadata['consumed_stop_request_sha256']=sha256_file(stop_file)
                stop_file.unlink()
            scorer_path=directory/'scorers'/f'update_{start:08d}'
            if not scorer_path.exists():
                from .scoring import export_scorer
                export_scorer(scorer_path,state,contract,normalization=prepared['normalization'],
                    adapter_contract=prepared['adapter_contract'],replicated=config['runtime']['device_count']>1)
        require(start<=config['training']['optimizer_updates'],'checkpoint exceeds configured budget')
        if start==config['training']['optimizer_updates']:
            summary=dict(status='COMPLETED',backend=BACKEND,upstream_commit=UPSTREAM_COMMIT,
                optimizer_updates=start,critic_optimizer_updates=start,value_optimizer_updates=start,
                actor_optimizer_updates=0,actor_loaded=False,sample_draws=draws,checkpoint=str(resume),
                elapsed_seconds=0.,already_complete_no_parameter_update=True,adapter=prepared['adapter_contract'],
                held_out_count=prepared['val_count'],scoring_api='official_idql.core.score_actions on encoded64 actions')
            atomic_json(directory/'summary.json',summary);atomic_json(directory/'status.json',summary)
            return summary
        directory.mkdir(parents=True,exist_ok=True)
        atomic_json(directory/'contract.json',contract);atomic_json(directory/'normalization.json',prepared['normalization'])
        (directory/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
        atomic_json(directory/'provenance.json',dict(backend=BACKEND,upstream=prepared['upstream'],contract=contract,
            frozen_behavior=config['frozen_behavior'],actor_loaded=False,actor_updates=0,
            runtime_environment={key:os.environ.get(key) for key in
                ('JAX_PLATFORMS','CUDA_VISIBLE_DEVICES','XLA_PYTHON_CLIENT_PREALLOCATE')}))
        stopped={'value':False};handlers={}
        def request_stop(*_):stopped['value']=True
        for kind in (signal.SIGINT,signal.SIGTERM):handlers[kind]=signal.signal(kind,request_stop)
        parallel=config['runtime']['device_count']>1
        update_fn=jax.pmap(update,axis_name='devices',devices=devices) if parallel else update
        began=time.perf_counter();metadata['segments']+=1;checkpoint=None
        try:
            for number in range(start+1,config['training']['optimizer_updates']+1):
                tick=time.perf_counter();indices=generator.integers(0,len(native),size=config['training']['global_batch_size'])
                batch=native.batch(indices)
                if parallel:
                    batch=jax.tree_util.tree_map(lambda x:x.reshape(len(devices),config['training']['per_device_batch_size'],*x.shape[1:]),batch)
                state,values=update_fn(state,batch)
                require(np.asarray(values['finite']).all(),'nonfinite parameters/optimizer/loss; refusing checkpoint')
                require(np.all(np.asarray(state.step)==number),'JAX optimizer update counter differs')
                draws+=len(indices)
                row=dict(optimizer_update=number,critic_optimizer_updates=number,value_optimizer_updates=number,
                    actor_optimizer_updates=0,sample_draws=draws,global_batch_size=len(indices),
                    process_segment=metadata['segments'],elapsed_seconds=time.perf_counter()-began,
                    update_seconds=time.perf_counter()-tick,
                    **{key:float(np.asarray(value).mean()) for key,value in values.items() if key not in {'finite','updates'}})
                require(all(math.isfinite(v) for v in row.values() if type(v) is float),'nonfinite metric')
                final=number==config['training']['optimizer_updates']
                requested=stopped['value'] or (directory/'stop.request').exists() or stop_after is not None and number>=stop_after
                validation=None
                if number%config['training']['validation_interval']==0 or final or requested:
                    single=jax.tree_util.tree_map(lambda x:x[0],state) if parallel else state
                    validation=evaluate(single,heldout,config['training']['validation_batch_size'])
                    _append_csv(directory/'validation.csv',{'optimizer_update':number,**validation})
                _append_csv(directory/'metrics.csv',row)
                with (directory/'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps({**row,**({'validation':validation} if validation else {})},allow_nan=False)+'\n')
                    stream.flush();os.fsync(stream.fileno())
                if number%config['training']['log_interval']==0 or number==start+1 or final:
                    print('OFFICIAL_IDQL_UPDATE '+json.dumps(row,allow_nan=False),flush=True)
                if number%config['training']['checkpoint_interval']==0 or final or requested:
                    checkpoint=save_checkpoint(directory/'checkpoints'/f'update_{number:08d}',state,contract,
                        dict(rng_state=generator.bit_generator.state,draws=draws,completed_updates=number),metadata)
                    from .scoring import export_scorer
                    export_scorer(directory/'scorers'/f'update_{number:08d}',state,contract,
                        normalization=prepared['normalization'],adapter_contract=prepared['adapter_contract'],replicated=parallel)
                atomic_json(directory/'status.json',dict(status='RUNNING',**row))
                if requested or final:break
            summary=dict(status='COMPLETED' if final else 'STOPPED',backend=BACKEND,upstream_commit=UPSTREAM_COMMIT,
                optimizer_updates=number,critic_optimizer_updates=number,value_optimizer_updates=number,
                actor_optimizer_updates=0,actor_loaded=False,sample_draws=draws,checkpoint=str(checkpoint),
                elapsed_seconds=time.perf_counter()-began,adapter=prepared['adapter_contract'],
                held_out_count=prepared['val_count'],scoring_api='official_idql.core.score_actions on encoded64 actions')
            atomic_json(directory/'summary.json',summary);atomic_json(directory/'status.json',summary)
            return summary
        except BaseException as exc:
            atomic_json(directory/'status.json',dict(status='FAILED',error=type(exc).__name__+': '+str(exc),actor_updates=0))
            raise
        finally:
            for kind,handler in handlers.items():signal.signal(kind,handler)
