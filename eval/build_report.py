"""Build a self-contained per-query HTML report for eval verification/debugging.

Joins three sources into one row-per-query view (Q01..Q48):
  - results_retrieval.jsonl : per-system {rr, rk}, stored as per-type positional `raw` arrays
  - results_generation.jsonl: LLM-judge scores (faithfulness/correctness/completeness) + reason
  - generated_answers.jsonl : question / gold / generated answer / retrieved context

Retrieval rows carry no query id, so the link is positional within each type, in
testset order: definition -> Q01..Q12, enumeration -> Q13..Q24, etc.

Run:  python eval/build_report.py   ->   eval/report.html  (open in a browser)
"""

import json
import html
from pathlib import Path

EVAL = Path(__file__).parent
TYPE_ORDER = ["definition", "enumeration", "indirect", "negation"]
SYSTEMS = ["dense", "bm25", "hybrid", "reranked"]


def load_jsonl(name):
    return [json.loads(line) for line in (EVAL / name).open(encoding="utf-8")]


def build_rows():
    testset = {r["id"]: r for r in load_jsonl("eval_testset_v1.jsonl")}
    answers = {r["id"]: r for r in load_jsonl("generated_answers.jsonl")}
    judge = {r["id"]: r for r in load_jsonl("results_generation.jsonl")}

    # Keep only the latest date per system, then map per-type raw -> query id by position.
    retr_raw = load_jsonl("results_retrieval.jsonl")
    latest = {}
    for d in retr_raw:
        key = d["system"]
        if key not in latest or d["date"] >= latest[key]["date"]:
            latest[key] = d

    # retrieval[qid][system] = {"rr":, "rk":}
    retrieval = {}
    for system, d in latest.items():
        for qtype in TYPE_ORDER:
            block = d["per_type"][qtype]
            start = TYPE_ORDER.index(qtype) * 12  # 12 questions per type
            for i, cell in enumerate(block["raw"]):
                qid = f"Q{start + i + 1:02d}"
                retrieval.setdefault(qid, {})[system] = cell

    rows = []
    for n in range(1, 49):
        qid = f"Q{n:02d}"
        ts = testset.get(qid, {})
        ans = answers.get(qid, {})
        jg = judge.get(qid, {})
        rows.append({
            "id": qid,
            "type": ts.get("type", ""),
            "question": ts.get("question", ""),
            "gold": ts.get("answer", ""),
            "answer": ans.get("result", ""),
            "context": ans.get("context", ""),
            "retrieval": {s: retrieval.get(qid, {}).get(s, {}) for s in SYSTEMS},
            "faithfulness": jg.get("faithfulness"),
            "correctness": jg.get("correctness"),
            "completeness": jg.get("completeness"),
            "reason": jg.get("reason", ""),
        })
    return rows


def render(rows):
    data = json.dumps(rows)
    tpl = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LocalRAG eval — per-query report</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--mut:#8b93a3;--line:#262b36;--ok:#2ea043;--mid:#d29922;--bad:#f85149;--txt:#e6e9ef}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--txt)}
header{padding:16px 20px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:5}
h1{font-size:16px;margin:0 0 10px}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
input,select{background:var(--card);color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:6px 9px;font-size:13px}
.pill{padding:3px 8px;border-radius:999px;font-size:11px;background:#222732;color:var(--mut)}
table{width:100%;border-collapse:collapse}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);cursor:pointer;user-select:none;white-space:nowrap}
tr.q{cursor:pointer}tr.q:hover{background:#1d212b}
td.q{font-variant-numeric:tabular-nums;color:var(--mut)}
.qtext{max-width:480px}
.s{display:inline-block;width:20px;text-align:center;border-radius:4px;font-size:11px;font-weight:600;margin-right:2px}
.hit{background:rgba(46,160,67,.18);color:var(--ok)}.miss{background:rgba(248,81,73,.16);color:var(--bad)}.na{color:var(--mut)}
.score{font-variant-numeric:tabular-nums;font-weight:600}
.g-ok{color:var(--ok)}.g-mid{color:var(--mid)}.g-bad{color:var(--bad)}
.detail{background:#10131a}.detail td{padding:0}
.panel{padding:14px 18px;display:grid;grid-template-columns:1fr 1fr;gap:14px}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.box h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
.box.full{grid-column:1/-1}
pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12px/1.5 ui-monospace,Consolas,monospace;color:#cdd3df}
.muted{color:var(--mut)}.count{color:var(--mut);font-size:12px;margin-left:8px}
</style></head><body>
<header>
<h1>LocalRAG eval — per-query report <span class="count" id="count"></span></h1>
<div class="controls">
<input id="q" placeholder="search question / answer / reason…" style="min-width:260px">
<select id="type"><option value="">all types</option></select>
<select id="filt">
<option value="">all rows</option>
<option value="any-miss">retrieval: any system missed</option>
<option value="all-miss">retrieval: all systems missed</option>
<option value="rerank-miss">retrieval: reranked missed</option>
<option value="low-corr">judge: correctness &lt; 0.8</option>
<option value="low-faith">judge: faithfulness &lt; 0.8</option>
</select>
<span class="pill">click a row to expand</span>
</div></header>
<table id="t"><thead><tr>
<th data-k="id">Q</th><th data-k="type">type</th><th class="qtext" data-k="question">question</th>
<th data-k="rdense">dns</th><th data-k="rbm25">bm</th><th data-k="rhybrid">hyb</th><th data-k="rreranked">rrk</th>
<th data-k="faithfulness">faith</th><th data-k="correctness">corr</th><th data-k="completeness">compl</th>
</tr></thead><tbody id="b"></tbody></table>
<script>
const DATA = __DATA__;
const SYS = ["dense","bm25","hybrid","reranked"];
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const rr = r => r && r.rk!=null;          // has a retrieval result
const hit = r => r && r.rk===1;           // gold chunk in top-k
function gcls(v){if(v==null)return "muted";return v>=0.9?"g-ok":v>=0.75?"g-mid":"g-bad";}
function scell(r){if(!rr(r))return '<span class="s na">·</span>';
  return '<span class="s '+(hit(r)?'hit':'miss')+'" title="rr='+r.rr+' rk='+r.rk+'">'+(hit(r)?'✓':'✗')+'</span>';}
const typeSel=document.getElementById("type");
[...new Set(DATA.map(d=>d.type))].forEach(t=>{let o=document.createElement("option");o.value=o.textContent=t;typeSel.append(o);});

let sortK="id",sortAsc=true;
function val(d,k){
  if(k.startsWith("r")&&SYS.includes(k.slice(1))){let c=d.retrieval[k.slice(1)];return rr(c)?(hit(c)?1:0):-1;}
  return d[k];
}
function passes(d){
  const q=document.getElementById("q").value.toLowerCase();
  const ty=typeSel.value, f=document.getElementById("filt").value;
  if(ty&&d.type!==ty)return false;
  if(q&&!((d.question+" "+d.answer+" "+d.reason+" "+d.gold).toLowerCase().includes(q)))return false;
  const miss=SYS.map(s=>d.retrieval[s]).filter(rr).filter(c=>!hit(c)).length;
  const present=SYS.map(s=>d.retrieval[s]).filter(rr).length;
  if(f==="any-miss"&&miss===0)return false;
  if(f==="all-miss"&&!(present>0&&miss===present))return false;
  if(f==="rerank-miss"&&!(rr(d.retrieval.reranked)&&!hit(d.retrieval.reranked)))return false;
  if(f==="low-corr"&&!(d.correctness!=null&&d.correctness<0.8))return false;
  if(f==="low-faith"&&!(d.faithfulness!=null&&d.faithfulness<0.8))return false;
  return true;
}
function row(d){
  return '<tr class="q" data-id="'+d.id+'">'+
    '<td class="q">'+d.id+'</td><td>'+esc(d.type)+'</td>'+
    '<td class="qtext">'+esc(d.question)+'</td>'+
    SYS.map(s=>'<td>'+scell(d.retrieval[s])+'</td>').join("")+
    ['faithfulness','correctness','completeness'].map(k=>
      '<td class="score '+gcls(d[k])+'">'+(d[k]==null?'·':d[k].toFixed(2))+'</td>').join("")+
    '</tr>';
}
function detail(d){
  const ret=SYS.map(s=>{let c=d.retrieval[s];return s+': '+(rr(c)?(hit(c)?'hit (rr='+c.rr+')':'MISS'):'n/a');}).join('  |  ');
  return '<tr class="detail"><td colspan="10"><div class="panel">'+
    '<div class="box"><h4>Question</h4><pre>'+esc(d.question)+'</pre></div>'+
    '<div class="box"><h4>Retrieval (top-k hit per system)</h4><pre>'+esc(ret)+'</pre></div>'+
    '<div class="box"><h4>Gold answer</h4><pre>'+esc(d.gold)+'</pre></div>'+
    '<div class="box"><h4>Generated answer</h4><pre>'+esc(d.answer)+'</pre></div>'+
    '<div class="box full"><h4>Judge reason — F '+d.faithfulness+' / C '+d.correctness+' / Cmpl '+d.completeness+'</h4><pre>'+esc(d.reason)+'</pre></div>'+
    '<div class="box full"><h4>Retrieved context</h4><pre>'+esc(d.context)+'</pre></div>'+
    '</div></td></tr>';
}
function draw(){
  let rows=DATA.filter(passes);
  rows.sort((a,b)=>{let x=val(a,sortK),y=val(b,sortK);if(x<y)return sortAsc?-1:1;if(x>y)return sortAsc?1:-1;return 0;});
  document.getElementById("b").innerHTML=rows.map(row).join("");
  document.getElementById("count").textContent="("+rows.length+" / "+DATA.length+" queries)";
}
document.getElementById("b").addEventListener("click",e=>{
  const tr=e.target.closest("tr.q");if(!tr)return;
  const nxt=tr.nextElementSibling;
  if(nxt&&nxt.classList.contains("detail")){nxt.remove();return;}
  document.querySelectorAll("tr.detail").forEach(x=>x.remove());
  const d=DATA.find(x=>x.id===tr.dataset.id);
  tr.insertAdjacentHTML("afterend",detail(d));
});
document.querySelectorAll("th").forEach(th=>th.addEventListener("click",()=>{
  const k=th.dataset.k;if(sortK===k)sortAsc=!sortAsc;else{sortK=k;sortAsc=true;}draw();}));
["q","type","filt"].forEach(id=>document.getElementById(id).addEventListener("input",draw));
draw();
</script></body></html>"""
    return tpl.replace("__DATA__", data)


def main():
    rows = build_rows()
    out = EVAL / "report.html"
    out.write_text(render(rows), encoding="utf-8")
    n_judged = sum(r["correctness"] is not None for r in rows)
    print(f"Wrote {out}  ({len(rows)} queries, {n_judged} judged)")


if __name__ == "__main__":
    main()
