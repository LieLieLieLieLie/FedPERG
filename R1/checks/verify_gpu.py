"""Verify complete CUDA experiments, checkpoints and frozen source provenance."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from fedperg.paths import DEVELOPMENT, FORMAL, PROTOCOLS, ROOT
from reporting.manuscript_tables import EXPECTED
from reporting.optimization_results import load


def main():
    records=load()
    counts=Counter(r['_suite'] for r in records)
    assert dict(counts)==EXPECTED,(dict(counts),EXPECTED)
    protocol=json.loads((PROTOCOLS/'r1_opt_confirmation_combined_gpu_protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():
        assert sha256((ROOT/'fedperg'/name).read_bytes()).hexdigest()==digest,name
    inventory=[]
    certificate_counts={'tensor_rounds':0,'temporal_rounds':0,'routing_decisions':0}
    all_records=[(Path(r['_path']),r) for r in records]
    for path in DEVELOPMENT.rglob('*.json'):
        payload=json.loads(path.read_text())
        assert payload['status']=='complete'
        all_records.append((path,payload))
    assert len(all_records)==sum(EXPECTED.values())+234
    for path,record in all_records:
        assert str(record['device']).startswith('cuda'),path
        assert record['config']['device']=='cuda',path
        assert record.get('peak_gpu_memory_mb',0)>0,path
        checkpoint=path.with_suffix('.pth')
        assert checkpoint.is_file() and checkpoint.stat().st_size>1000,checkpoint
        if record['config']['method']=='FedPERG':
            assert path.with_name(path.stem+'__operator.pth').is_file(),path
        assert record['history'][-1]['round']==record['config']['rounds'],path
        assert len(record['diagnostics'])==record['config']['rounds'],path
        assert all(np.isfinite(float(v)) for k,v in record['final'].items()
                   if isinstance(v,(int,float))),path
        for diagnostic in record['diagnostics']:
            if diagnostic.get('skipped'):continue
            if record['config'].get('evidence_control')=='informative':
                for value in diagnostic.get('utility_fitted_linear_effects',[]):
                    assert value<=1e-10,(path,value)
                    certificate_counts['tensor_rounds']+=1
                if 'utility_momentum_linear_effect' in diagnostic:
                    assert diagnostic['utility_momentum_linear_effect']<=1e-10,path
                    certificate_counts['temporal_rounds']+=1
            losses=diagnostic.get('selector_expert_losses')
            if losses and len(losses)==7:
                selected=diagnostic['selector_expert_index']
                router=min(losses[i] for i in [0,1,3,5])
                assert losses[selected]<=router+1e-8,path
                if selected in [2,4,6]:
                    assert losses[selected]+diagnostic['selector_margin']<router,path
                certificate_counts['routing_decisions']+=1
        inventory.append({'path':str(path.relative_to(ROOT)),
                          'json_sha256':sha256(path.read_bytes()).hexdigest(),
                          'model_sha256':sha256(checkpoint.read_bytes()).hexdigest()})
    assert not list(FORMAL.rglob('*.tmp')),'Incomplete checkpoint commits remain'
    data_dir=Path(records[0]['config']['data_dir'])
    caches=['cifar10_resnet18_ftta_v1.npz','cifar100_resnet18_ftta_v1.npz',
            'officehome65_resnet18.npz','officehome_raw_48.npz']
    result={'formal_count':sum(counts.values()),'suite_counts':dict(counts),
            'development_count':234,'device':'cuda','records':inventory,
            'environment':{'python':__import__('sys').version,'torch':torch.__version__,
                           'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)},
            'data_cache_sha256':{name:sha256((data_dir/name).read_bytes()).hexdigest() for name in caches},
            'source_sha256':{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                             for p in sorted(ROOT.rglob('*.py')) if 'results' not in p.parts},
            'certificate_and_fallback_checks':certificate_counts,
            'github_updated':False}
    (PROTOCOLS/'gpu_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Verified',sum(counts.values()),'formal and 234 development CUDA runs and their checkpoints.')


if __name__=='__main__':
    main()
