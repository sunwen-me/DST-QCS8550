#!/usr/bin/env python3
"""Resume DST-Calib evaluation network conversion/linking with QAIRT 2.50 only."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",type=Path,required=True)
    root=p.parse_args().model_dir.resolve()
    import qai_hub as hub
    source=root/"eva_sb.onnx"
    manifest=json.loads((root/"export_manifest.json").read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if digest!=manifest["onnx_sha256"]:raise RuntimeError("ONNX digest mismatch")
    state_path=root/"qairt250_jobs.json"
    state=json.loads(state_path.read_text()) if state_path.exists() else {"onnx_sha256":digest}
    if state["onnx_sha256"]!=digest:raise RuntimeError("stale jobs for another ONNX graph")
    def save():
        temp=state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(state,indent=2)+"\n");temp.replace(state_path)
    def wait(job,stage):
        previous=None
        while True:
            status=job.get_status();name=getattr(status.state,"name",str(status.state))
            if name!=previous:
                print(stage,job.job_id,name,flush=True);previous=name
                state[stage+"_status"]=name;save()
            if name=="SUCCESS":return
            if name in ("FAILED","ERROR","CANCELED"):raise RuntimeError(str(status))
            time.sleep(10)
    if "compile_job" not in state:
        job=hub.submit_compile_job(model=str(source),device=hub.Device("QCS8550 (Proxy)"),
            name="dst_eva_sb_256x512_qairt250",options="--target_runtime qnn_dlc --qairt_version 2.50")
        state.update(compile_job=job.job_id,compile_options=job.options);save()
    else:job=hub.get_job(state["compile_job"])
    wait(job,"compile")
    if "link_job" not in state:
        link=hub.submit_link_job(models=[job.get_target_model()],device=hub.Device("QCS8550 (Proxy)"),
            name="dst_eva_sb_256x512_qairt250_link",options="--qairt_version 2.50")
        state.update(link_job=link.job_id,link_options=link.options);save()
    else:link=hub.get_job(state["link_job"])
    wait(link,"link")
    target=link.get_target_model();metadata={str(k):str(v) for k,v in target._metadata.items()}
    if not any(v.startswith("2.50.") for v in metadata.values()):raise RuntimeError("not a 2.50 context")
    path=root/"eva_sb_qairt250.bin"
    if not path.exists():Path(target.download(str(path.with_suffix(".partial.bin")))).replace(path)
    state.update(context_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),context_bytes=path.stat().st_size,
        target_model_id=target.model_id,target_metadata=metadata);save()
    manifest["qnn"]={"context_sha256":state["context_sha256"],"metadata":metadata}
    (root/"export_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(state,indent=2),flush=True)


if __name__=="__main__":main()
