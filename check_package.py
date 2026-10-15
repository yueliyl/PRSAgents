"""Check the complete student release without needing Ollama or test labels."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from harness import Dataset


def main(root, full=False):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    for item in manifest['files']:
        path=root/item['path']
        assert path.is_file(),path
        assert path.stat().st_size==item['bytes'],path
        if full:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    paths=sorted((root/'data/datasets').glob('dataset_*/rep*'))
    assert len(paths)==40,f'Expected 40 replicates, found {len(paths)}'
    forbidden=('labels_test','regime.json','_instructor','founders.npz','alias_map')
    for path in root.rglob('*'):
        assert not any(word in path.relative_to(root).as_posix() for word in forbidden),path
    for path in paths:
        d=Dataset(str(path))
        assert d.dosage.shape==(4000,1000),path
        assert d.S.shape==(4000,10) and d.C.shape==(4000,12),path
        assert (len(d.idx_train),len(d.idx_val),len(d.idx_test))==(2000,800,1200),path
        assert np.isnan(d.y[d.idx_test]).all(),path
        assert set(d.idx_train).isdisjoint(d.idx_val),path
        assert set(d.idx_train).isdisjoint(d.idx_test),path
        assert set(d.idx_val).isdisjoint(d.idx_test),path
        assert 'oracle_test_r2' not in d.info,path
        assert np.isfinite(d.S).all() and np.isfinite(d.C).all(),path
        with np.load(path/'stage1/sumstats.npz') as ss:
            assert ss['block_of_snp'].shape==(1000,),path
        with np.load(path/'stage1/eval_phenotypes.npz') as ev:
            assert all(ev[f'y_{pop}'].shape==(1500,) for pop in ev['populations']),path
    print(f'PASS: {len(paths)} datasets; fixed splits, dimensions and withheld test labels verified')
    print(f'PASS: {len(manifest["files"])} manifest files'+(' with SHA256 checks' if full else ' checked'))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=Path(__file__).resolve().parent)
    ap.add_argument('--full',action='store_true')
    a=ap.parse_args(); main(a.root,a.full)
