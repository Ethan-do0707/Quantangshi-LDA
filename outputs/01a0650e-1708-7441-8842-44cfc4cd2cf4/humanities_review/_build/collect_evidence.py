"""只读取冻结结果并建立来源清单，不重新进行统计分析。"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
LIT = ROOT / 'outputs/01a05061-9ee3-7610-a401-abfec81de1b2/literary_history_dataset'
PRE = ROOT / 'outputs/01a05cec-c8db-7761-9760-9d2d42a4853e'
BCD = ROOT / 'outputs/01a0628d-ef13-7961-85ce-146e0a14f60a/BCD_finding_level_robustness'
AUD = ROOT / 'historical_metadata/representativeness_audit'

def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

paths = set()
for folder in [BCD, AUD, PRE / 'author_period_trend_structure', PRE / 'author_period_effect_validation', PRE / 'author_influence_loao']:
    paths.update((folder / 'data').glob('*.csv'))
    paths.update(folder.glob('*Report.md'))
    paths.update(folder.glob('*manifest.json'))
paths.update((ROOT / 'historical_metadata/master').glob('*.csv'))
for folder in [LIT / 'taxonomy', LIT / 'alignment', LIT / 'alignment/alignment_review']:
    paths.update(folder.glob('*.csv'))
    paths.update(folder.glob('*.md'))
paths.add(ROOT / 'outputs/01a05061-9ee3-7610-a401-abfec81de1b2/model_selection/final_model_selection_report.md')
for name in ['effect_validation', 'author_influence', 'BCD_robustness', 'trend_structure']:
    paths.add(ROOT / f'config/{name}_config.yaml')
paths.update((LIT / 'freeze').rglob('*.npy'))
paths.update((LIT / 'freeze').rglob('*.npz'))
manifest = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)} for p in sorted(paths)]
(OUT / '_build/source_hashes_before.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
locations = {
    'comparison': BCD / 'data/BCD_finding_robustness_comparison.csv',
    'effects': BCD / 'data/BCD_finding_effects.csv',
    'mapping': BCD / 'data/finding_semantic_mapping.csv',
    'ladder': BCD / 'data/finding_robustness_ladder.csv',
    'taxonomy': LIT / 'taxonomy/C_K20_topic_taxonomy.csv',
    'topic_metadata': LIT / 'taxonomy/topic_metadata.csv',
    'alignment': LIT / 'alignment/BCD_cross_model_topic_alignment.csv',
    'validation': PRE / 'author_period_effect_validation/data/clear_topic_primary_contrasts.csv',
    'candidates': PRE / 'author_period_effect_validation/data/statistically_supported_candidate_trends.csv',
    'loao': PRE / 'author_influence_loao/data/author_validated_candidate_trends.csv',
    'exclusion': PRE / 'author_influence_loao/data/top_contributor_exclusion_effects.csv',
    'contributions': PRE / 'author_influence_loao/data/author_period_topic_contributions.csv',
    'priority_author': BCD / 'data/priority_author_sensitivity.csv',
    'coverage': AUD / 'data/period_metadata_coverage_audit.csv',
    'smd': AUD / 'data/topic_probability_representativeness.csv',
    'risk': AUD / 'data/clear_topic_representativeness.csv',
    'samples': AUD / 'data/analysis_sample_representativeness.csv',
}
data = {k:rows(p) for k,p in locations.items()}
authors = rows(ROOT / 'historical_metadata/master/QTS_author_historical_metadata.csv')
master = rows(ROOT / 'historical_metadata/master/C_K20_literary_history_master_with_author_history.csv')
data['master_verification'] = {'authors':len(authors), 'poems':len(master), 'poem_date_nonempty':sum(bool(r['date'] or r['year']) for r in master), 'master_sha256':digest(ROOT / 'historical_metadata/master/C_K20_literary_history_master_with_author_history.csv')}
data['representative_metadata'] = {}
for t in data['taxonomy']:
    if t['topic_id'] in ['topic_03','topic_06','topic_08','topic_16']:
        ids = [t[f'representative_poem_{i}'].split(' | ')[0] for i in range(1,6)]
        data['representative_metadata'][t['topic_id']] = [{k:r[k] for k in ['poem_id','author','title','author_period_primary','cbdb_match_confidence']} for r in master if r['poem_id'] in ids]
(OUT / '_build/evidence.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'source_files':len(manifest), 'master':data['master_verification'], 'coverage':data['coverage'], 'representative_metadata':data['representative_metadata'], 'top_authors':[{k:r[k] for k in ['topic_id','author','period','poem_count','share_of_period_topic_mass']} for r in data['contributions'] if r['topic_id'] in ['topic_03','topic_06','topic_08'] and r['rank_by_topic_mass']=='1'], 'mapping':[{k:r[k] for k in ['finding_id','target_model','target_topic_labels','semantic_notes','human_review_status']} for r in data['mapping'] if r['finding_id'] in ['T03_EL','T06_EL','T08_EL']]}, ensure_ascii=False, indent=2))
