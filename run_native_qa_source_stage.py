"""Verify the complete execution-input seal around each bounded study stage."""
import hashlib
import json
from pathlib import Path

import native_qa_source_study as study


def sealed_inputs():
    metadata=study.verify()
    binding=study.read(study.OUT/'input_binding.json')
    paths={name:expected for name,expected in binding['additional_dependencies'].items()}
    for name,expected in paths.items():assert study.sha(study.ROOT/name)==expected,name
    for seed in (47,83):
        folder=study.OUT/f'seed{seed}'
        pre=study.read(folder/'preflight.json')
        assert pre['passed'] and pre['experiment_sha256']==study.sha(study.OUT/'frozen_experiment.json')
        for path in [folder/'packets.pt',folder/'preflight.json']+[folder/arm/'initial/model.pt' for arm in study.ARMS]:
            paths[str(path.relative_to(study.ROOT))]=study.sha(path)
    for path in [Path(__file__).resolve(),study.OUT/'input_binding.json',study.OUT/'frozen_experiment.json']:
        paths[str(path.relative_to(study.ROOT))]=study.sha(path)
    for result_path in study.OUT.glob('seed*/*/source_results.json'):
        result=study.read(result_path)
        assert study.sha(result_path.parent/'final/model.pt')==result['checkpoint_sha256'],result_path
    target=study.OUT/'execution_input_seal.json'
    value=dict(learning_updates_at_seal=0,files=paths,scope='Exact frozen feature packets, full replay inputs including styles, execution preflight, initial arm checkpoints, runner, schedules, and guards.')
    if target.exists():assert study.read(target)==value,'Execution input seal changed'
    else:study.dump(target,value)
    return target


if __name__=='__main__':
    seal=sealed_inputs()
    study.main()
    # Every completed stage must leave its frozen execution inputs untouched.
    assert sealed_inputs()==seal
