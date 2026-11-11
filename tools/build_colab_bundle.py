"""Package only required source/data files, and bind the Colab notebook to the ZIP."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(output=None, update_notebook=True):
    output=Path(output or ROOT/'outputs/colab_distillation_bundle.zip')
    output.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    for directory in ('distillation_loop','self_instruct_pipeline'):
        files.extend(sorted((ROOT/directory).glob('*.py')))
    files.extend(ROOT/name for name in (
        'requirements-colab.txt','config/self_instruct.json','config/distillation_loop.json',
        'data/self_instruct/seed_tasks.jsonl','data/self_instruct/task_schemas.json',
        'data/self_instruct/GENERATION_PROMPT.md',
        'data/evaluation/fixed_validation_v1.jsonl','data/evaluation/fixed_validation_v1.manifest.json',
    ))
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            # Fixed timestamps make this portable artifact reproducible.
            info=zipfile.ZipInfo('Health-Insurance-SLM-Colab/'+path.relative_to(ROOT).as_posix(),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            archive.writestr(info,path.read_bytes())
    checksum=hashlib.sha256(output.read_bytes()).hexdigest()
    if update_notebook:
        notebook=ROOT/'distillation_colab_100.ipynb'
        nb=json.loads(notebook.read_text())
        for cell in nb['cells']:
            source=''.join(cell.get('source',[]))
            source=re.sub(r'EXPECTED_BUNDLE_SHA256 = "[A-Za-z0-9_]+"',f'EXPECTED_BUNDLE_SHA256 = "{checksum}"',source)
            cell['source']=source.splitlines(True)
        notebook.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n')
    print('Colab bundle:',output,'SHA256:',checksum)
    return output


if __name__=='__main__':
    build()
