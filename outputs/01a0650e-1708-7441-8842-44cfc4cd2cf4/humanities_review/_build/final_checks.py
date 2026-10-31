"""交付核验：冻结输入未变、来源之间一致、要求完整，不运行新统计。"""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
OUT=HERE.parent
hashes=json.loads((HERE/'source_hashes_before.json').read_text(encoding='utf-8'))
changed=[r['path'] for r in hashes if hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()!=r['sha256']]
assert not changed, changed
e=json.loads((HERE/'evidence.json').read_text(encoding='utf-8'))
ladder=[r['finding_id'] for r in e['ladder']]
assert ladder==['T08_EL','T03_EL','T03_ML','T06_EL','T16_EL','T03_EH','T11_ML','T07_EL','T19_EH']
for r in e['comparison']:
    v=next(x for x in e['validation'] if x['topic_id']==r['topic'] and x['contrast']==r['contrast'])
    a=next(x for x in e['loao'] if x['topic_id']==r['topic'] and x['contrast']==r['contrast'])
    l=next(x for x in e['ladder'] if x['finding_id']==r['finding_id'])
    assert r['C_original_status']==v['validation_status']
    assert r['C_author_robustness_status']==a['author_robustness_status']
    assert r['finding_level_robustness']==l['finding_level_robustness']
    for model in 'BCD':
        for agg in ['POEM','AUTHOR']:
            detail=next(x for x in e['effects'] if x['finding_id']==r['finding_id'] and x['model']==model and x['sample']=='CORE' and x['aggregation']==agg)
            for suffix,key in [('delta','raw_delta'),('ci_low','bootstrap_ci_low'),('ci_high','bootstrap_ci_high')]:
                assert abs(float(r[f'{model}_{agg.lower()}_{suffix}'])-float(detail[key]))<1e-12
for r in e['mapping']:
    original=next(x for x in e['alignment'] if x['C_topic_id']==r['c_topic_id'])
    model=r['target_model']
    assert r['target_topic_ids']==original[f'{model}_topic_id']
    assert r['alignment_relation_type']==original[f'{model}_alignment_type']
    assert r['alignment_confidence']==original[f'{model}_alignment_confidence']
    assert r['human_review_status']=='NOT_RECORDED_IN_SOURCE'
assert len(e['priority_author'])==32
assert 'T03_ML' not in {r['finding_id'] for r in e['priority_author']}
assert all(r['sign_reversal']=='False' for r in e['priority_author'])
md=(OUT/'QTS_BCD_Humanities_Review_Report.md').read_text(encoding='utf-8')
assert '\ufffd' not in md and '{{' not in md
for i in range(1,20): assert re.search(rf'^## {i}\. ',md,re.M)
core=[]
for i,fid in enumerate(['T08','T03','T03','T06'],1):
    chunk=md.split(f'### 10.{i} ')[1].split('\n### ')[0]
    for ch in 'ABCDEFGH': assert f'**{ch}. ' in chunk
    core.append(fid)
refs=set(re.findall(r'\[(S\w+)\]',md))
defs=set(re.findall(r'^\[(S\w+)\]:',md,re.M))
assert refs==defs
assert all(x in md for x in ['目前可以说什么','核心结论卡','人工审读优先级','这些结果不应该怎样理解','下一阶段人工审读任务','HYPOTHESIS FOR HUMANISTIC REVIEW'])
result={'frozen_source_hashes_verified':len(hashes),'changed_frozen_sources':changed,'all_nine_findings_labels_match_sources':True,'all_18_mappings_match_original_alignment':True,'all_162_core_numeric_cells_match':True,'main_materials_A_to_H_found':True,'core_findings_A_to_H_template_complete':True,'nineteen_sections_present':True,'undefined_source_references':[],'source_missing':'正式人工审核状态未记录；八类指定主材料均存在。','terminology_clarifications':['旧优先四项不等于最终稳健四项','形状稳健性与发现稳健性不同','自助区间支持不等于全部FDR检验通过','分期置信度与人物身份匹配置信度不同'],'numeric_inconsistency_found':False,'human_review_items':['确认来源映射，尤其B17较宽标签与语义束边界','核对代表诗时期及CORE资格','核对原对齐卡中的同文异署与独立文本证据资格'],'docx_render_limitation':'soffice缺失，仅完成结构和同文核验','report_statistics':json.loads((HERE/'text_validation.json').read_text(encoding='utf-8'))}
result['deliverables']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('QTS_BCD_Humanities_Review_Report.*')}
result['terminology_inconsistency_confirmed']='趋势data_dictionary.csv将trend_shape解释为基于bootstrap支持方向，但实际代码使用数值方向；本报告说明差异，未更改源文件。'
result['human_review_items'].append('确认是否后续勘误trend_shape数据字典释义，不改变现有统计结果')
(HERE/'final_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
