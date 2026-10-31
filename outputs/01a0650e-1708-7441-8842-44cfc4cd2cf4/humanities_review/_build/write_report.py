"""将已冻结的CSV取值填入审读文本；仅做转录和一致性核验。"""
from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE.parent
data = json.loads((HERE / 'evidence.json').read_text(encoding='utf-8'))
lit = 'outputs/01a05061-9ee3-7610-a401-abfec81de1b2/literary_history_dataset'
pre = 'outputs/01a05cec-c8db-7761-9760-9d2d42a4853e'
bcd = 'outputs/01a0628d-ef13-7961-85ce-146e0a14f60a/BCD_finding_level_robustness'
aud = 'historical_metadata/representativeness_audit'
sources = {
 'S0': 'outputs/01a05061-9ee3-7610-a401-abfec81de1b2/model_selection/final_model_selection_report.md',
 'S1t': f'{lit}/taxonomy/C_K20_topic_taxonomy.csv',
 'S1m': f'{lit}/taxonomy/topic_metadata.csv',
 'S2a': 'historical_metadata/master/QTS_author_historical_metadata.csv',
 'S2m': 'historical_metadata/master/C_K20_literary_history_master_with_author_history.csv',
 'S3r': f'{aud}/Period_Metadata_Representativeness_Audit_Report.md',
 'S3c': f'{aud}/data/period_metadata_coverage_audit.csv',
 'S3p': f'{aud}/data/author_productivity_known_vs_unknown.csv',
 'S3s': f'{aud}/data/topic_probability_representativeness.csv',
 'S3k': f'{aud}/data/clear_topic_representativeness.csv',
 'S3a': f'{aud}/data/analysis_sample_representativeness.csv',
 'S3l': f'{aud}/data/period_metadata_sample_labels.csv',
 'S4r': f'{pre}/author_period_trend_structure/QTS_Author_Period_Trend_Structure_Report.md',
 'S4dict': f'{pre}/author_period_trend_structure/data/data_dictionary.csv',
 'S4code': 'src/run_author_period_trend_structure.py',
 'S5r': f'{pre}/author_period_effect_validation/QTS_Author_Period_Effect_Size_Statistical_Validation_Report.md',
 'S5c': f'{pre}/author_period_effect_validation/data/clear_topic_primary_contrasts.csv',
 'S5s': f'{pre}/author_period_effect_validation/data/statistically_supported_candidate_trends.csv',
 'S5cfg': 'config/effect_validation_config.yaml',
 'S6r': f'{pre}/author_influence_loao/QTS_Author_Influence_LOAO_Report.md',
 'S6c': f'{pre}/author_influence_loao/data/author_validated_candidate_trends.csv',
 'S6x': f'{pre}/author_influence_loao/data/top_contributor_exclusion_effects.csv',
 'S6a': f'{pre}/author_influence_loao/data/author_period_topic_contributions.csv',
 'S7r': f'{bcd}/QTS_BCD_Finding_Level_Robustness_Report.md',
 'S7c': f'{bcd}/data/BCD_finding_robustness_comparison.csv',
 'S7e': f'{bcd}/data/BCD_finding_effects.csv',
 'S7m': f'{bcd}/data/finding_semantic_mapping.csv',
 'S7l': f'{bcd}/data/finding_robustness_ladder.csv',
 'S7u': f'{bcd}/data/BCD_common_universe_qc.csv',
 'S7p': f'{bcd}/data/priority_author_sensitivity.csv',
 'S8': f'{lit}/alignment/BCD_cross_model_topic_alignment.csv',
 'S8c03': f'{lit}/alignment/alignment_review/C_topic_03_alignment.md',
 'S9b': 'human_topic_review/B_K20',
 'S9c': 'human_topic_review/C_K20',
 'S9d': 'human_topic_review/D_K20',
}
for key, p in sources.items():
    assert (ROOT / p).exists(), (key,p)
catalog = []
groups = [
 ('A. 主模型与主题命名', ['S1t','S1m']),
 ('B. 历史元数据', ['S2a','S2m']),
 ('C. 元数据代表性', ['S3r','S3c','S3p','S3s','S3k','S3a','S3l']),
 ('D. 趋势结构', ['S4r','S4dict','S4code']),
 ('E. 效应量与统计验证', ['S5r','S5c','S5s','S5cfg']),
 ('F. 作者影响与逐作者删除', ['S6r','S6c','S6x','S6a']),
 ('G. 发现层跨模型稳健性', ['S7r','S7m','S7c','S7e','S7l','S7u','S7p']),
 ('H. 原跨模型对齐', ['S8','S8c03']),
 ('补充. 原模型选择与代表诗资料目录', ['S0','S9b','S9c','S9d']),
]
for label, keys in groups:
    catalog.append(f'**{label}。**')
    for key in keys:
        catalog.append(f'- [{key}] {Path(sources[key]).name}')
    catalog.append('')
replacements = {'source_catalog': '\n'.join(catalog)}
for n in ['03','06','08']:
    tax = next(t for t in data['taxonomy'] if t['topic_id']==f'topic_{n}')
    replacements[f'words_{n}'] = tax['top_20_words'].replace(';','、')

def table(header, rows):
    return '\n'.join(['| '+' | '.join(header)+' |', '|'+ '|'.join(['---']*len(header))+'|'] + ['| '+' | '.join(map(str,r))+' |' for r in rows])

core_ids = ['T08_EL','T03_EL','T03_ML','T06_EL']
precise = []
checks = []
for fid in core_ids:
    r=next(x for x in data['comparison'] if x['finding_id']==fid)
    assert r['finding_level_robustness']=='ROBUST_ACROSS_BCD'
    assert r['sample_direction_consistency']=='B:6/6;C:6/6;D:6/6'
    t=r['topic']; con=r['contrast']
    former=next(x for x in data['validation'] if x['topic_id']==t and x['contrast']==con)
    author=next(x for x in data['loao'] if x['topic_id']==t and x['contrast']==con)
    assert former['validation_status']==r['C_original_status']
    assert author['author_robustness_status']==r['C_author_robustness_status']
    tab=[]
    for model in 'BCD':
        row=next(x for x in data['effects'] if x['finding_id']==fid and x['sample']=='CORE' and x['aggregation']=='POEM' and x['model']==model)
        row_a=next(x for x in data['effects'] if x['finding_id']==fid and x['sample']=='CORE' and x['aggregation']=='AUTHOR' and x['model']==model)
        for agg,detail in [('poem',row),('author',row_a)]:
            for suffix,key in [('delta','raw_delta'),('ci_low','bootstrap_ci_low'),('ci_high','bootstrap_ci_high')]:
                assert abs(float(r[f'{model}_{agg}_{suffix}'])-float(detail[key]))<1e-12
                checks.append(f'{fid}:{model}:{agg}:{suffix}')
            assert r[f'{model}_{agg}_bootstrap'] == detail['bootstrap_supported_direction']
        p=float(r[f'{model}_poem_delta']); a=float(r[f'{model}_author_delta'])
        support = 'P/A均支持上升' if p>0 else 'P/A均支持下降'
        mapping='C原主题（锚点）' if model=='C' else r[f'{model}_mapping_type']+'（HIGH）'
        tab.append([model,f'{p:+.3f}',f'{a:+.3f}',support,mapping])
        display=fid[:3]+' '+con
        ci=lambda agg:'['+f"{float(r[f'{model}_{agg}_ci_low']):+.6f}, {float(r[f'{model}_{agg}_ci_high']):+.6f}"+']'
        precise.append([display,model,f'{p:+.6f}',ci('poem'),f'{a:+.6f}',ci('author')])
    replacements[f'core_{fid}']=table(['模型','POEM Δ','AUTHOR Δ','bootstrap方向','mapping类型'],tab)
    ex=[x for x in data['exclusion'] if x['topic_id']==t and x['contrast']==con and x['sample']=='CORE' and x['exclusion_scheme']=='GLOBAL_TOP' and x['N'] in ['1','3','5']]
    values={}
    for agg in ['POEM','AUTHOR']:
        values[agg]=[next(x for x in ex if x['aggregation']==agg and x['N']==n) for n in ['1','3','5']]
        assert all(x['sign_flip']=='False' for x in values[agg])
    pv='、'.join(f"{float(x['new_delta']):+.3f}" for x in values['POEM'])
    av='、'.join(f"{float(x['new_delta']):+.3f}" for x in values['AUTHOR'])
    replacements[f'exclusion_{fid}']=f'按GLOBAL_TOP依次排除Top1、Top3、Top5作者后，POEM差值分别约{pv}，AUTHOR约{av}，方向均保持。'
replacements['precise_table']=table(['Finding','模型','P Δ','P 95% CI','A Δ','A 95% CI'],precise)

glossary = [
 ('LDA','潜在狄利克雷分配，是从文档中的词语共现识别主题混合的统计模型。它提供可审读结构，不自行形成文学史解释。'),
 ('Topic','模型学习出的词语—语义共现结构。人工名称是解释入口，不是传统诗类的确定边界。'),
 ('Topic probability','每首诗分配给某Topic的相对权重；二十项总和约为1。不等于主题判断准确率、相应字数比例或某类诗首数。'),
 ('dominant topic','一首诗权重最高的Topic。仅据它硬分类会丢失次级主题与混合语义。'),
 ('Topic taxonomy / Topic type','冻结的主题命名、类型与使用范围。CLEAR中心清楚，BROAD范围宽，MIXED语义混合，NOISE缺乏可靠解释中心，均非诗歌价值等级。'),
 ('TF','一个词在整个语料中出现的总次数。重复出现在同一首诗中也重复计次。'),
 ('DF','一个词出现于多少首不同诗。一首诗中重复出现仍只计一首。'),
 ('K=20','模型设置二十个Topic；不表示唐诗在文学史上只有二十种题材。'),
 ('B_K20','宽词表对照模型，最低TF=2、DF=2，保护词按既定规则例外保留。'),
 ('C_K20','中等过滤的主模型，最低TF=3、DF=3；本报告T编号以它为锚点。'),
 ('D_K20','严格过滤对照模型，最低TF=5、DF=3；用于检验词表变化下的发现稳定性。'),
 ('author period','作者所属时期。归属标签继承到诗级用于分组，但不是该诗创作年代。'),
 ('poem composition date / poem date','单篇作品的实际或考定创作年代。当前母表的逐诗日期字段未提供这种信息。'),
 ('POEM-weighted','每首诗等权，反映入选现存文本总体的平均结构；诗作记录多的作者贡献较大。'),
 ('AUTHOR-balanced','先取作者内均值，再让作者等权，反映入选作者群体平均结构；不等于中位数或多数作者的状态。'),
 ('Topic profile / estimand','Topic profile是一位作者或一首诗的主题权重概况；estimand是希望估计的量。POEM与AUTHOR的估计对象不同，因此都须保留。'),
 ('effect size','效应量，回答差异有多大。大小需结合样本与语义理解，不能自动转换成文学重要性。'),
 ('raw delta / Raw Δ','后期均值减前期均值的主题概率绝对差。E-L按晚减初计算，+0.070为增加7个百分点。'),
 ("Hedges' g",'以作者间合并波动尺度衡量两个时期均值距离，并作小样本修正。本项目提供作者级g，其大小等级只是统计参照。'),
 ("Cliff's delta / Cliff's δ",'后期作者主题值高于前期作者的配对概率，减去低于的概率。它关注排序，不是原始权重差。'),
 ('bootstrap','在现有样本中有放回重抽，再重复计算同一指标。它检查作者组合带来的不确定性，不搜集新证据。'),
 ('95% CI','95%置信区间，用于表达差值的不确定范围。本报告核心表采用已有自助法区间，跨零表示方向支持不足，不表示必定没有差异。'),
 ('cluster bootstrap','以作者为一簇抽样，作者的全部入选诗一同进入计算。这样避免把同一作者多首诗当成完全独立样本。'),
 ('bootstrap-supported direction','区间全在零以上为支持上升，全在零以下为支持下降，跨零为不确定。它与原始差值正负号分开保存。'),
 ('p-value','在检验假设成立时取得当前或更极端结果的概率，不是结论出错的概率。很小也不能说明差异很重要。'),
 ('Omnibus test','四个时期整体差异的总体检验。它不直接定位某两个时期的差异。'),
 ('planned contrast / E-H / H-M / M-L / E-L','预先规定的两期比较；E、H、M、L依次为初、盛、中、晚唐作者组，差值均为后组减前组。'),
 ('FDR','错误发现率；多次检验时，对被判显著结果中误报的期望比例加以控制。它不能保证每个具体结论正确。'),
 ('BH correction / q','Benjamini–Hochberg是本项目的FDR校正方法，q为对应的校正p值。它与自助法区间属于不同的支持判据。'),
 ('SMD','标准化均值差，把两组均值差放到数据自身通常波动的尺度中。已知与未知时期两组的平均绝对SMD约0.041，指检查到的整体平均主题差异很小。'),
 ('LOAO','逐一暂时移除作者及其入选诗，检查同一对比的方向与幅度。它检验单个作者对结果的支撑程度。'),
 ('Top contributor exclusion','依主题概率总量贡献排序，移除前1、3、5等若干作者。用来检查少数高贡献作者共同支撑结果的可能。'),
 ('Topic mass','作者各首诗的某主题概率之和，是本项目贡献排名的量。它不是某传统诗类的诗作首数。'),
 ('GLOBAL_TOP / PER_PERIOD_TOP','前者在比较涉及的两时期合并排名删除，后者每期分别排名删除。同样的N对应不同总删除人数，不可择优替换主分类。'),
 ('AUTHOR_ROBUST','C中既定作者删除检查下方向与幅度稳定，且未触发高集中度标签；不表示对任何作者删除方式都稳健。'),
 ('CONCENTRATED_BUT_ROBUST','主题贡献集中，但通过既定逐作者及Top1/3/5检查；仍可发生幅度衰减，也不保证删后所有区间均不跨零。'),
 ('AUTHOR_SENSITIVE','移除关键作者后反向或强烈衰减等，触发作者敏感规则。不能作为时代整体趋势的强证据。'),
 ('Topic alignment','以关键词、代表诗与词分布相似性判断跨模型可比语义。编号相同没有对应意义，来源高置信不等于人工签署确认。'),
 ('DIRECT','一个C主题与一个目标主题直接对应。可比性较清楚，但仍不保证两者语义范围完全相同。'),
 ('split','一个C语义场在另一模型中分成多个主题。不能只因编号或相似词就自行认定分裂。'),
 ('BUNDLE / semantic bundle','依既有split把目标主题概率无权相加，恢复较完整语义范围。它不是与C完全相同的潜在变量。'),
 ('single-best','只取来源中最相似的一个目标主题作敏感性对照。它与语义束结果不同，可能提示结论依赖语义范围。'),
 ('merge','一个目标主题容纳多个C主题的语义，原焦点语义可能无法单独隔离。'),
 ('MERGED_PROXY','以合并后的较宽目标作语义代理。即使方向支持，也因比较对象不够纯粹而限制稳健等级。'),
 ('ROBUST_ACROSS_BCD','具体时期对比通过三词表条件的冻结稳健性规则。它是方法证据，不是文学史确认。'),
 ('PARTIALLY_ROBUST','有部分支持，但区间、映射、样本、聚合或语义束依赖等限制尚在。不等于结果错误。'),
 ('MODEL_SENSITIVE','达到既定跨模型方向敏感规则的分类。本批为0项，不能拿它统称所有部分稳健结果。'),
 ('FULL_AGREEMENT / 6/6','前者是发现层CORE方向支持模式的一致标签；后者指三样本×两聚合的六个原始差值同向。二者均不能扩展成所有相邻时期或所有区间均一致。'),
 ('representativeness risk','样本对更大语料范围的代表性风险，结合主题差异等多项诊断。低或中风险都不是无偏抽样的证明。'),
 ('STRICT / CORE / EXPANDED','严格、核心与扩展的冻结作者时期样本，兼顾分期置信度、跨期边界与覆盖率。不能与身份匹配分数混用。'),
 ('KNOWN / UNKNOWN','当前作者时期已知或未知；UNKNOWN记录可用于不依赖时期的主题描述，不能自动填入某个时期。'),
 ('CLR','中心化对数比，将某主题权重与该诗整体主题构成的几何平均水平比较后取对数。只用于尺度敏感性检查，不替代原概率。'),
 ('compositional data','组成数据，各主题权重总和约为1。某部分增加是相对份额变化，不自动意味着某类诗绝对数量增加。'),
 ('Semantic Drift / Internal Change','同一Topic在不同时期的内部语义、场景或表达关系可能改变。当前只列为待检验问题，尚无细读结论。'),
 ('STATISTICALLY_SUPPORTED_CANDIDATE / LITERARY_HISTORY_CONFIRMED','前者是达到原统计规则的候选，后者意味着文学史解释已经确认。当前不得将前者或Level 4改写成后者。'),
]
replacements['glossary']='\n\n'.join(f'**{term}。** {meaning}' for term,meaning in glossary)
md=(HERE/'report_template.md').read_text(encoding='utf-8')
for key,value in replacements.items():
    assert '{{'+key+'}}' in md, key
    md=md.replace('{{'+key+'}}',value)
assert '{{' not in md
md+='\n\n'+'\n'.join(f'[{key}]: <{(ROOT / p).as_posix()}>' for key,p in sources.items())+'\n'
visible=re.sub(r'^\[S\w+\]:.*$', '', md, flags=re.M)
visible=re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',visible)
visible=re.sub(r'(?m)^#{1,6}\s*|\*\*|`|(?m:^[>|]\s*)','',visible)
opening=md.split('## 1. 给人文学者的阅读说明\n\n')[1].split('\n## 2.')[0]
count={'chinese_characters':len(re.findall(r'[\u3400-\u9fff]',visible)), 'visible_nonwhitespace_characters':len(re.sub(r'\s','',visible)), 'chinese_characters_in_opening':len(re.findall(r'[\u3400-\u9fff]',opening)), 'opening_nonwhitespace':len(re.sub(r'\s','',opening)), 'glossary_entries':len(glossary), 'top_level_sections':len(re.findall(r'^## \d+\.',md,re.M)), 'core_numeric_cross_checks':len(checks), 'all_source_links_exist':True}
assert 800 <= count['chinese_characters_in_opening'] <= 1200, count
assert count['top_level_sections']==19
(OUT/'QTS_BCD_Humanities_Review_Report.md').write_text(md,encoding='utf-8')
(HERE/'text_validation.json').write_text(json.dumps(count,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(count,ensure_ascii=False,indent=2))
