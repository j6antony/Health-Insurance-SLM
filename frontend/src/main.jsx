import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const label = value => String(value ?? '').replaceAll('_', ' ');
const Badge = ({children, type='warn'}) => <span className={`badge ${type}`}>{children}</span>;
async function request(url, options) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Request failed. Please retry.');
  return body;
}
function Rule({rule}) {
  const text = rule.operation ? (rule.operation === 'OR' ? 'Any one alternative is enough' : 'All of these are required') : label(rule.required_document || rule.required);
  return <div className="rule"><strong>{text}</strong> <Badge type={rule.satisfied ? 'good' : 'warn'}>{rule.satisfied ? 'Found' : 'Not found'}</Badge>
    {rule.required_fields && <p>Required fields in the same document: {rule.required_fields.map(label).join(', ')}</p>}
    {rule.children?.map((child,i) => <Rule key={i} rule={child}/>)}</div>;
}
function Results({data}) {
  const {documents, verification, requirements} = data;
  const conflicts = verification.results.filter(r => r.status === 'conflict').length;
  const review = documents.some(d => d.classification.requires_review || d.quality?.status === 'review_required' || d.extraction.fields.some(f => f.requires_review));
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const link = document.createElement('a'); link.href=url; link.download='claim-verification.json'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  return <section aria-labelledby="result-title"><div className="result-heading"><h2 id="result-title">Verification results</h2><button className="secondary" onClick={download}>Download JSON</button></div>
    <div className="metrics"><div className="metric"><span>DOCUMENTS PROCESSED</span><strong>{documents.length}</strong></div><div className="metric"><span>DOCUMENTATION RULE</span><strong>{requirements.satisfied ? 'Satisfied' : 'Not satisfied'}</strong></div><div className="metric"><span>REVIEW STATUS</span><strong>{conflicts ? `${conflicts} conflicts` : review ? 'Review needed' : 'No flags'}</strong></div></div>
    <section className="card"><h3>Required documents & fields</h3><Rule rule={requirements}/>{review && <p>Some classifications or fields need review. Confirm their evidence before relying on a satisfied requirement.</p>}</section>
    <section className="card"><h3>Cross-document checks</h3>{verification.results.map(r => <details className="check" key={r.field_name}><summary><strong>{label(r.field_name)}</strong><Badge type={r.status==='consistent'?'good':r.status==='conflict'?'bad':'warn'}>{label(r.status)}</Badge></summary>{r.observations.length ? r.observations.map((o,i)=><p key={i}><strong>{documents.find(d=>d.document_id===o.document_id)?.filename}</strong> · Page {o.page_number}<br/>{o.value}<br/>Evidence: {o.evidence}</p>) : <p>No populated value found. At least two documents with this field are needed for comparison.</p>}</details>)}</section>
    <h3>Documents & extracted fields</h3>{documents.map(d=><article className="card" key={d.document_id}><h3 className="document-title">{d.filename}</h3><Badge type={d.classification.requires_review?'warn':'good'}>{label(d.classification.document_type)}</Badge> {d.classification.requires_review && <Badge>Classification needs review</Badge>}<p>{d.page_count} pages · {d.extraction.fields.length} extracted fields</p>
      {!!d.classification.evidence?.length && <details><summary>Classification evidence</summary>{d.classification.evidence.map((e,i)=><p className="evidence" key={i}>{e}</p>)}</details>}
      {d.quality?.status==='review_required' && <p>Extraction quality needs review: {d.quality.issues?.map(label).join(', ')}</p>}
      {d.extraction.fields.length ? <div className="table-wrap"><table><thead><tr><th>Field</th><th>Value & source evidence</th><th>Status</th></tr></thead><tbody>{d.extraction.fields.map((f,i)=><tr key={i}><td>{label(f.field_name)}</td><td><strong>{f.value}</strong><div className="evidence">Page {f.page_number} · {f.evidence}</div></td><td><Badge type={f.requires_review?'warn':'good'}>{f.requires_review?'Review':'Extracted'}</Badge></td></tr>)}</tbody></table></div> : <p>No populated fields found. Blank form labels do not count as values.</p>}</article>)}
  </section>;
}
function App() {
  const [files,setFiles]=useState([]), [busy,setBusy]=useState(false), [error,setError]=useState(''), [result,setResult]=useState(null), [seconds,setSeconds]=useState(0);
  const input=useRef(null);
  useEffect(()=>{if(!busy)return; const timer=setInterval(()=>setSeconds(s=>s+1),1000);return()=>clearInterval(timer);},[busy]);
  function addFiles(incoming) {
    if(busy)return;
    const next=[...files,...Array.from(incoming)];
    if(next.some(f=>!f.name.toLowerCase().endsWith('.pdf')||!f.size))return setError('Choose nonempty PDF files only.');
    if(next.length>10||next.reduce((s,f)=>s+f.size,0)>30*1024*1024)return setError('Choose up to 10 PDFs totaling at most 30 MB.');
    setFiles(next);setError('');
  }
  async function verify() {
    if(busy||!files.length)return;
    setBusy(true);setSeconds(0);setError('');setResult(null);
    try {
      const form=new FormData();files.forEach(f=>form.append('files',f));
      const {job_id}=await request('/api/verify',{method:'POST',body:form});
      let failures=0;
      while(true) {
        await new Promise(resolve=>setTimeout(resolve,1500));
        let job;
        try {job=await request(`/api/jobs/${job_id}`);failures=0;}catch(e){if(++failures<4)continue;throw e;}
        if(job.status==='failed')throw new Error(job.error);
        if(job.status==='complete'){setResult(job.result);break;}
      }
    }catch(e){setError(e.message||'Verification failed. Please retry.');}finally{setBusy(false);}
  }
  return <main><header><span className="brand">✚ &nbsp; Claim check</span><span className="local">Local workspace</span></header><section className="intro"><p className="eyebrow">DOCUMENT VERIFICATION</p><h1>Check your claim package.</h1><p>Upload your documents to check required information and compare details across files.</p></section>
    <section className="card"><h2>Upload documents</h2><p>PDF only · Up to 10 files · 30 MB total</p><label className="drop" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();addFiles(e.dataTransfer.files);}}><span className="upload-icon">↑</span><strong>Choose PDFs or drop them here</strong><span>Include all documents for one claim.</span><input ref={input} type="file" accept="application/pdf,.pdf" multiple disabled={busy} onChange={e=>{addFiles(e.target.files);e.target.value='';}}/></label>
    <ul id="file-list">{files.map((f,i)=><li key={i}><span>{f.name} · {(f.size/1024/1024).toFixed(2)} MB</span><button className="secondary" disabled={busy} aria-label={`Remove ${f.name}`} onClick={()=>setFiles(files.filter((_,j)=>j!==i))}>×</button></li>)}</ul><button id="verify" disabled={busy||!files.length} onClick={verify}>Verify package <span>→</span></button><p role="status" aria-live="polite">{busy ? `Processing… ${seconds}s elapsed. Local model processing may take several minutes.` : result ? 'Verification complete. Review results below.' : ''}</p>{error && <p id="error" role="alert">{error}</p>}</section>
    <details className="rules"><summary>What does this check?</summary><p>The hospitalization rule accepts any one of: a UB-04 form; a physician statement with a diagnosis; or both room and board documentation and medical codes documentation.</p><p>Patient name, policy number, hospital, and admission/discharge dates are compared across documents. Some required document types are not yet supported by the classifier. Uncertain results need review. This checks documentation, not claim approval.</p></details>
    {result && <Results data={result}/>}<footer>Processed locally with Ollama. Temporary uploads are deleted after processing. Results remain in backend memory until eviction or restart.</footer></main>;
}
createRoot(document.getElementById('root')).render(<App/>);
