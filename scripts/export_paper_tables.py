from __future__ import annotations
import json
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
ROOT=Path(__file__).resolve().parents[1]
METRICS=ROOT/'results/metrics/numeric_metrics.jsonl'
OUTPUT=ROOT/'results/tables/TrustStatAgent_ESWA_tables.xlsx'
def avg(groups,d,v,k):
    z=[float(x[k]) for x in groups.get((d,v),[]) if x.get(k) is not None]
    return sum(z)/len(z) if z else '-'
def pc(groups,d,v,k):
    x=avg(groups,d,v,k); return '-' if x=='-' else 100*x
def add(wb,name,headers,rows):
    ws=wb.create_sheet(name); ws.append(headers)
    for r in rows: ws.append(r)
    fill=PatternFill('solid',fgColor='D9EAF7')
    for c in ws[1]: c.font=Font(bold=True); c.fill=fill; c.alignment=Alignment(horizontal='center')
    for col in ws.columns: ws.column_dimensions[col[0].column_letter].width=min(max(len(str(x.value or '')) for x in col)+2,34)
    ws.freeze_panes='A2'
def main():
    rows=[json.loads(x) for x in METRICS.read_text().splitlines() if x.strip()]
    g={}
    for x in rows:g.setdefault((x['dataset'],x['variant']),[]).append(x)
    F='TrustStatAgent'; C='ReAct with Conventional Gating'; N='w/o Conformal Calibration'; GATE='w/o Risk Gate'; CA='California Housing'; CO='Carbon Monitor China/Total'
    wb=Workbook(); wb.remove(wb.active)
    add(wb,'Table1_Task',['Dataset','Task','Metric','TrustStatAgent'],[
      ['StatQA','Statistical QA','Official Accuracy (%)','-'],['Adult','Binary classification','Accuracy (%)',pc(g,'Adult',F,'accuracy')],['California Housing','Regression','MAE',avg(g,CA,F,'mae')],['Carbon Monitor','One-day forecasting','MAE',avg(g,CO,F,'mae')]])
    add(wb,'Table1_StatQA_Baselines',['Method','Official Accuracy (%)'],[['Zero-shot-CoT','-'],['ReAct Statistical Agent','-'],['ReAct + Conventional Gate','-'],['TrustStatAgent','-']])
    base=[]
    for label,d in [('Adult','Adult'),('California',CA),('Carbon',CO)]:
      base += [[label,'ReAct Statistical Agent','-',100,0,pc(g,d,GATE,'error_release_rate')],
               [label,'ReAct + Conventional Gate','-',pc(g,d,C,'release_rate'),pc(g,d,C,'manual_review_rate'),pc(g,d,C,'error_release_rate')],
               [label,'TrustStatAgent',pc(g,d,F,'coverage'),pc(g,d,F,'release_rate'),pc(g,d,F,'manual_review_rate'),pc(g,d,F,'error_release_rate')]]
    add(wb,'Table2_Baseline_Reliability',['Dataset','Method','Coverage (%)','Auto-release (%)','Manual-review (%)','Error-release (%)'],base)
    add(wb,'Table3_Planning_Ablation',['Variant','Official Accuracy (%)'],[['w/o Explicit Planning','-'],['Full TrustStatAgent','-']])
    ab=[]
    for label,d in [('Adult','Adult'),('California',CA),('Carbon',CO)]:
      ab += [[label,N,pc(g,d,N,'coverage'),pc(g,d,N,'release_rate'),pc(g,d,N,'manual_review_rate'),pc(g,d,N,'error_release_rate')],
             [label,GATE,'-',100,0,pc(g,d,GATE,'error_release_rate')],
             [label,F,pc(g,d,F,'coverage'),pc(g,d,F,'release_rate'),pc(g,d,F,'manual_review_rate'),pc(g,d,F,'error_release_rate')]]
    add(wb,'Table4_Module_Ablation',['Dataset','Variant','Coverage (%)','Auto-release (%)','Manual-review (%)','Error-release (%)'],ab)
    notes=wb.create_sheet('Notes'); notes.append(['Status'])
    for s in ['StatQA values are - until the final full formal run is complete.','Numeric values are latest three-seed test averages.','ReAct Statistical Agent means direct release after successful tool execution.','Conventional Gate uses raw probability/interval/residual thresholds.','Error-release is the error rate among automatically released outputs; a separate Selective Risk column would duplicate it.','Do not replace - with pilot values.'] : notes.append([s])
    notes.column_dimensions['A'].width=120
    OUTPUT.parent.mkdir(parents=True,exist_ok=True); wb.save(OUTPUT); print(OUTPUT)
if __name__=='__main__': main()
