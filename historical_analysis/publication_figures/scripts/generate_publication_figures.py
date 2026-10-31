"""从冻结CSV生成论文图；不重新计算统计结果。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PERIODS = ['初唐', '盛唐', '中唐', '晚唐']
CODES = ['E', 'H', 'M', 'L']
TOPICS = ['topic_03', 'topic_06', 'topic_08']
CORE = ['T03_EL', 'T03_ML', 'T06_EL', 'T08_EL']
ORDER = ['T08_EL', 'T03_EL', 'T03_ML', 'T06_EL', 'T16_EL', 'T03_EH', 'T11_ML', 'T07_EL', 'T19_EH']
AGG = {
    'POEM': dict(color='#0072B2', ls='-', marker='o', label='按诗等权（POEM）'),
    'AUTHOR': dict(color='#B45A25', ls='--', marker='s', label='作者均衡（AUTHOR）'),
}
MODEL = {
    'B': dict(color='#0072B2', ls='--', marker='^', label='B：较宽过滤'),
    'C': dict(color='#00856A', ls='-', marker='o', label='C：中等过滤'),
    'D': dict(color='#8B659E', ls=':', marker='D', label='D：较严过滤'),
}
TOPIC_STYLE = {
    'topic_03': dict(color='#4D4D4D', ls='-', marker='o'),
    'topic_06': dict(color='#0072B2', ls='--', marker='s'),
    'topic_08': dict(color='#B45A25', ls=':', marker='^'),
}
MAPPING = {
    'topic_03': ('topic_08;topic_19', 'topic_17;topic_08', 'BUNDLE', 'B08+B19 ↔ C03 ↔ D17+D08'),
    'topic_06': ('topic_14;topic_13', 'topic_05;topic_11', 'BUNDLE', 'B14+B13 ↔ C06 ↔ D05+D11'),
    'topic_08': ('topic_17', 'topic_12', 'DIRECT', 'B17 ↔ C08 ↔ D12'),
    'topic_16': ('topic_06', 'topic_19', 'DIRECT', 'B06 ↔ C16 ↔ D19'),
}
TIME_NOTE = '时期依据作者所属时期，而非单篇作品创作年代；初、盛、中、晚唐为有序类别，不表示连续年份。'
BCD_NOTE = 'B、C、D分别为较宽、中等和较严格词汇过滤设置；三模型的 Topic probability 绝对尺度并非完全相同，因此本图主要比较效应方向与稳健性，而非要求三个数值完全一致。'
CI_NOTE = '误差线直接采用趋势阶段已保存的95%时期分层作者簇 bootstrap CI（2,000次，seed=20260901）。'
EFFECT_NOTE = '误差线直接采用最终BCD表的95%时期分层作者簇 bootstrap CI（5,000次，seed=20260902）；Δ为后一作者时期均值减前一时期均值，零线表示无差值。'
SOURCE, TABLE, HASHES, NAMES = {}, {}, {}, {}
AUDIT, MANIFEST, CAPTIONS, LAYOUT = [], [], [], []
NOW = datetime.now().astimezone().isoformat(timespec='seconds')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check(label, condition, detail=''):
    AUDIT.append(dict(check=label, status='PASS' if condition else 'FAIL', detail=str(detail)))
    if not condition:
        raise RuntimeError(f'冻结结果核验失败，停止绘图：{label}；{detail}')


def close(label, actual, expected, tolerance=1e-12):
    error = float(np.max(np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))))
    check(label, error <= tolerance, f'max_abs_error={error:.3g}; tolerance={tolerance}')


def read(key, path):
    SOURCE[key] = path.resolve()
    HASHES[str(path.resolve())] = sha(path)
    data = pd.read_csv(path, float_precision='round_trip', encoding='utf-8-sig')
    data['source_file'] = str(path.resolve())
    data['source_row'] = np.arange(len(data)) + 2
    TABLE[key] = data


def locate_sources():
    groups = {
        'trend': ('QTS_Author_Period_Trend_Structure_Report.md', {'means': 'period_topic_means.csv', 'ci': 'period_topic_bootstrap_ci.csv', 'trend_clear': 'clear_topic_trend_structure.csv'}),
        'effect': ('QTS_Author_Period_Effect_Size_Statistical_Validation_Report.md', {'primary': 'clear_topic_primary_contrasts.csv'}),
        'influence': ('QTS_Author_Influence_LOAO_Report.md', {'candidates': 'author_validated_candidate_trends.csv', 'exclusion': 'top_contributor_exclusion_effects.csv', 'exclusion_ci': 'top_exclusion_bootstrap_ci.csv'}),
        'bcd': ('QTS_BCD_Finding_Level_Robustness_Report.md', {'effects': 'BCD_finding_effects.csv', 'comparison': 'BCD_finding_robustness_comparison.csv', 'ladder': 'finding_robustness_ladder.csv', 'mapping': 'finding_semantic_mapping.csv'}),
    }
    candidates = []
    for group, (report_name, files) in groups.items():
        valid = []
        for report in ROOT.glob(f'outputs/**/{report_name}'):
            manifest = report.parent / 'analysis_manifest.json'
            if not manifest.exists():
                continue
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            passed = meta.get('invariance_tests_passed') == meta.get('invariance_tests_total', -1)
            present = all((report.parent / 'data' / name).exists() for name in files.values())
            candidates.append(dict(group=group, report=str(report), qc_passed=passed, required_files_present=present, mtime=report.stat().st_mtime))
            if passed and present:
                valid.append(report)
        check(f'{group} 有完成且通过QC的来源', bool(valid))
        report = max(valid, key=lambda p: p.stat().st_mtime)
        meta_path = report.parent / 'analysis_manifest.json'
        SOURCE[group + '_report'], SOURCE[group + '_manifest'] = report, meta_path
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        for path in (report, meta_path):
            HASHES[str(path)] = sha(path)
        for key, name in files.items():
            path = report.parent / 'data' / name
            expected = meta['outputs'].get('data\\' + name, meta['outputs'].get('data/' + name))
            check(f'{name} 已登记来源哈希', expected is not None)
            check(f'{name} 与冻结manifest一致', sha(path) == expected['sha256'])
            read(key, path)
        for key, record in meta['inputs'].items():
            path = Path(record['path'])
            if str(path) not in HASHES:
                actual = sha(path)
                check(f'{group} 输入 {key} 哈希', actual == record['sha256'])
                HASHES[str(path)] = actual
    meta = json.loads(SOURCE['trend_manifest'].read_text(encoding='utf-8'))
    taxonomy = next(Path(item['path']) for item in meta['inputs'].values() if Path(item['path']).name == 'topic_metadata.csv')
    read('taxonomy', taxonomy)
    NAMES.update(TABLE['taxonomy'].set_index('topic_id').final_topic_label.to_dict())
    return candidates


def effects(findings, aggregation=None, model=None):
    data = TABLE['effects']
    mask = data.finding_id.isin(findings) & data['sample'].eq('CORE')
    if aggregation:
        mask &= data.aggregation.eq(aggregation)
    if model:
        mask &= data.model.eq(model)
    return data.loc[mask].copy()


def preflight():
    ladder = TABLE['ladder']
    check('最终阶梯顺序', ladder.finding_id.tolist() == ORDER)
    check('最终四项稳健finding', set(ladder.loc[ladder.finding_level_robustness.eq('ROBUST_ACROSS_BCD'), 'finding_id']) == set(CORE))
    check('其余五项部分稳健', ladder.finding_level_robustness.tolist()[4:] == ['PARTIALLY_ROBUST'] * 5)
    for row in TABLE['effects'].itertuples():
        key = f'{row.finding_id}/{row.model}/{row.specification}'
        check(key + ' 区间与方向', row.bootstrap_ci_low <= row.raw_delta <= row.bootstrap_ci_high and np.sign(row.raw_delta) == (1 if row.direction == 'POSITIVE' else -1))
        close(key + ' Δ端点', row.period_B_mean - row.period_A_mean, row.raw_delta)
    for topic, (b, d, kind, _) in MAPPING.items():
        for model, ids in (('B', b), ('D', d)):
            part = TABLE['mapping'].loc[lambda x: x.c_topic_id.eq(topic) & x.target_model.eq(model)]
            check(f'{topic}/{model} mapping', len(part) > 0 and set(part.analysis_topic_ids) == {ids} and set(part.mapping_testability) == {kind})
            part = TABLE['effects'].loc[lambda x: x.c_topic_id.eq(topic) & x.model.eq(model)]
            check(f'{topic}/{model} effect mapping', set(part.mapped_topic_ids) == {ids} and set(part.mapping_type) == {kind})
    for name in ('ladder', 'comparison'):
        for row in effects(ORDER).itertuples():
            expected = TABLE[name].loc[lambda x: x.finding_id.eq(row.finding_id)].iloc[0]
            prefix = f'{row.model}_{row.aggregation.lower()}'
            close(f'{name}/{row.finding_id}/{prefix}', [row.raw_delta, row.bootstrap_ci_low, row.bootstrap_ci_high], [expected[prefix + '_delta'], expected[prefix + '_ci_low'], expected[prefix + '_ci_high']], 0)
    text = SOURCE['bcd_report'].read_text(encoding='utf-8')
    report_checks = []
    for topic, contrast, section in re.findall(r'## \d+\. T(\d+) ([EHM]–[HML])\n(.*?)(?=\n## |\Z)', text, re.S):
        fid = f"T{topic}_{contrast.replace('–', '')}"
        for model, agg, delta, low, high in re.findall(r'\| ([BCD]) \| (POEM|AUTHOR) \| (-?\d+\.\d+) \| \[(-?\d+\.\d+), (-?\d+\.\d+)\]', section):
            row = effects([fid], agg, model).iloc[0]
            actual = [f'{row.raw_delta:.6f}', f'{row.bootstrap_ci_low:.6f}', f'{row.bootstrap_ci_high:.6f}']
            check(f'最终报告逐格核对 {fid}/{model}/{agg}', actual == [delta, low, high])
            report_checks.append(dict(finding_id=fid, model=model, aggregation=agg, raw_delta=row.raw_delta, ci_low=row.bootstrap_ci_low, ci_high=row.bootstrap_ci_high, report_delta=delta, report_ci_low=low, report_ci_high=high, status='PASS'))
    check('最终报告54行162数值完整核对', len(report_checks) == 54)
    pd.DataFrame(report_checks).to_csv(OUT / 'qa/report_numeric_crosscheck.csv', index=False, encoding='utf-8-sig')
    for topic in TOPICS:
        means = TABLE['means'].query("sample == 'CORE' and topic_id == @topic")
        for agg in AGG:
            four = means.loc[means.aggregation.eq(agg)].set_index('period').loc[PERIODS]
            bcd = effects(['T' + topic[-2:] + '_EL'], agg, 'C').iloc[0]
            close(f'{topic}/{agg} C原始与BCD四期均值', four.mean_topic_probability, bcd[['mean_' + c for c in CODES]])
            ci = TABLE['ci'].query("sample == 'CORE' and topic_id == @topic and aggregation == @agg").set_index('period').loc[PERIODS]
            close(f'{topic}/{agg} 均值与CI点估计', four.mean_topic_probability, ci['mean'], 0)
            check(f'{topic}/{agg} 四期CI有界', bool(((ci.ci_2_5 <= ci['mean']) & (ci['mean'] <= ci.ci_97_5) & (ci.ci_2_5 >= 0)).all()))
        check(topic + ' taxonomy名称', NAMES[topic] == means.topic_label.iloc[0])
    for fid in CORE + ['T16_EL']:
        for agg in AGG:
            row = effects([fid], agg, 'C').iloc[0]
            old = TABLE['primary'].loc[lambda x: x.topic_id.eq(row.c_topic_id) & x.contrast.eq(row.contrast)].iloc[0]
            close(f'{fid}/{agg} C统计验证差值', row.raw_delta, old['raw_effect_' + agg.lower()])
    t16 = effects(['T16_EL'])
    check('T16六个效应同向为负', bool((t16.raw_delta < 0).all()))
    b = t16.loc[t16.model.eq('B')]
    check('T16 B两种CI跨0', bool(((b.bootstrap_ci_low < 0) & (b.bootstrap_ci_high > 0)).all()))


def configure_font():
    preferred = ['Noto Serif CJK SC', 'Noto Serif SC', 'Noto Sans CJK SC', 'Noto Sans SC', 'Source Han Serif SC', 'Source Han Sans SC', 'SimSun', 'Microsoft YaHei']
    available = {item.name for item in fm.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), None)
    check('系统中文字体可用', chosen is not None)
    path = fm.findfont(fm.FontProperties(family=chosen))
    original_path = path
    source_family = chosen
    # 系统思源可变字体默认被识别为细体；实例化400字重供论文打印。
    if chosen in ('Noto Serif SC', 'Noto Sans SC'):
        from fontTools.ttLib import TTFont
        from fontTools.varLib.instancer import instantiateVariableFont
        font = TTFont(path)
        if 'fvar' in font:
            folder = OUT / 'qa/fonts'
            folder.mkdir(exist_ok=True)
            fixed = folder / 'QTS_Noto_Regular.ttf'
            if not fixed.exists():
                instantiateVariableFont(font, {'wght': 400}, inplace=True).save(fixed)
            path = str(fixed)
            chosen = 'QTS Noto Regular'
            fm.fontManager.ttflist.insert(0, fm.FontEntry(fname=path, name=chosen, style='normal', weight=400))
    plt.rcParams.update({'font.family': [chosen, 'DejaVu Sans'], 'font.size': 10.5, 'axes.labelsize': 11, 'axes.titlesize': 12.5, 'axes.titleweight': 'normal', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#777777', 'axes.linewidth': .6, 'xtick.color': '#343434', 'ytick.color': '#343434', 'text.color': '#222222', 'axes.labelcolor': '#222222', 'axes.unicode_minus': True, 'legend.frameon': False, 'legend.fontsize': 9.5, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path', 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    return dict(font_family=chosen, source_font_family=source_family, font_file=path, original_font_file=original_path, font_weight=400, font_sha256=sha(path))


def tid(topic):
    return 'T' + topic[-2:]


def heading(fig, title, subtitle):
    fig.text(.10, .955, title, fontsize=15, va='top')
    fig.text(.10, .906, subtitle, fontsize=10, color='#606060', va='top')


def foot(fig, text):
    fig.text(.10, .025, text, fontsize=9, va='bottom', color='#555555', linespacing=1.5)


def axis_style(ax, probability=False):
    ax.grid(axis='y', color='#E5E5E5', linewidth=.65)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    if probability:
        ax.set_xticks(range(4), PERIODS)
        ax.set_xlim(-.22, 3.22)
        ax.set_xlabel('作者所属时期', labelpad=9)
        ax.set_ylabel('平均 Topic probability', labelpad=10)
    else:
        ax.axhline(0, color='#696969', linewidth=.85, zorder=1)
        ax.set_ylabel('raw Δ（主题概率差）', labelpad=10)


def limits(data, low='bootstrap_ci_low', high='bootstrap_ci_high'):
    lo, hi = min(0, data[low].min()), max(0, data[high].max())
    gap = (hi - lo) * .14
    return lo - gap, hi + gap


def period_data(topic):
    data = TABLE['means'].query("sample == 'CORE' and topic_id == @topic").copy()
    ci = TABLE['ci'].query("sample == 'CORE' and topic_id == @topic")
    keys = ['sample', 'aggregation', 'period', 'topic_id']
    ci = ci[keys + ['ci_2_5', 'ci_97_5', 'bootstrap_iterations', 'random_seed', 'source_file', 'source_row']].rename(columns={'source_file': 'ci_source_file', 'source_row': 'ci_source_row'})
    data = data.merge(ci, on=keys, validate='one_to_one')
    data['model'], data['universe'] = 'C', 'C_ORIGINAL'
    return data.sort_values(['aggregation', 'period_order'])


def plot_period(ax, data, compact=False):
    for agg, style in AGG.items():
        rows = data.loc[data.aggregation.eq(agg)].set_index('period').loc[PERIODS]
        y = rows.mean_topic_probability.to_numpy()
        ax.errorbar(np.arange(4), y, yerr=[y - rows.ci_2_5, rows.ci_97_5 - y], color=style['color'], linestyle=style['ls'], marker=style['marker'], markerfacecolor=style['color'] if agg == 'POEM' else 'white', markersize=5, linewidth=1.5, elinewidth=.9, capsize=3, label=style['label'], zorder=3)
    axis_style(ax, True)
    ax.set_ylim(0, data.ci_97_5.max() * 1.12)
    if compact:
        ax.set_xlabel('')


def mapping_caption(topic):
    if topic in TOPICS[:2]:
        return f'语义映射为 {MAPPING[topic][3]}；B/D probability 为冻结split Topic概率的无权和（BUNDLE），并非天然一对一Topic。本阶段直接使用已保存的概率和，未重新合并。'
    return f'语义映射为 {MAPPING[topic][3]}，来源标记HIGH-confidence DIRECT；B/C/D中存在较明确的一对一语义对应。来源未记录正式human_review_status，此处不补称新的人工批准。'


def trend_caption(topic):
    text = f'图示C_K20 CORE样本中，{tid(topic)}“{NAMES[topic]}”按作者所属时期分组的平均Topic probability。蓝色实线圆点为POEM-weighted（每诗等权），棕色虚线空心方点为AUTHOR-balanced（先作者内均值，再对作者等权）。{CI_NOTE}{TIME_NOTE}纵轴从0起，单题图上限依据本题区间设置；不同题目的视觉斜率不宜直接比较。'
    if topic == 'topic_03':
        text += '初唐—晚唐（E-L）及中唐—晚唐（M-L）为ROBUST_ACROSS_BCD；初唐—盛唐（E-H）仅为PARTIALLY_ROBUST。四期均值保留盛唐至中唐的小幅回升，不将总体下降写成持续下降。'
    else:
        text += '初唐—晚唐（E-L）为ROBUST_ACROSS_BCD；标签针对该对比，不表示每个相邻时期对比均获得同等级支持。'
    if topic == 'topic_08':
        text += mapping_caption(topic)
    return text


def save(fig, base, figure_id, title, data, caption, usage, core=False, sources=(), alias_of=''):
    # 逐图保留完整精度与来源行号；输出前检查文本是否被画布裁切。
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    outside = []
    for txt in fig.findobj(matplotlib.text.Text):
        if txt.get_visible() and txt.get_text():
            box = txt.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.x1 + 1 or box.y1 > canvas.y1 + 1:
                outside.append(txt.get_text())
    check(base + ' 文本不超出画布', not outside, outside)
    LAYOUT.append(dict(figure=base, text_outside_canvas=len(outside), width_in=fig.get_figwidth(), height_in=fig.get_figheight()))
    data_file = OUT / 'data' / (base + '_plot_data.csv')
    data.to_csv(data_file, index=False, encoding='utf-8-sig')
    paths = {str(SOURCE[k]) for k in sources}
    paths.add(str(SOURCE['taxonomy']))
    for col in data.columns:
        if col.endswith('source_file'):
            paths.update(str(v) for v in data[col].dropna().unique() if v)
    topic_col = next((c for c in ('topic_id', 'c_topic_id', 'topic') if c in data), None)
    for suffix in ('png', 'pdf', 'svg'):
        path = OUT / suffix / (base + '.' + suffix)
        if suffix == 'png':
            fig.savefig(path, dpi=600)
        elif suffix == 'pdf':
            fig.savefig(path, metadata={'Title': title, 'Subject': '按作者所属时期分组；冻结结果可视化', 'Creator': 'Matplotlib; QTS publication figures'})
        else:
            fig.savefig(path)
        MANIFEST.append(dict(figure_id=figure_id, figure_filename=str(path.relative_to(OUT)), figure_title=title, source_files=json.dumps(sorted(paths), ensure_ascii=False), topic_id=';'.join(data[topic_col].dropna().astype(str).unique()) if topic_col else '', contrast=';'.join(data.contrast.dropna().astype(str).unique()) if 'contrast' in data else '四时期', sample='CORE;STRICT;EXPANDED' if base.startswith('Figure_07') else 'CORE', aggregation=';'.join(data.aggregation.dropna().unique()) if 'aggregation' in data else 'POEM;AUTHOR', models=';'.join(data.model.dropna().unique()) if 'model' in data else 'B;C;D', data_rows_used=len(data), plot_data_file=str(data_file.relative_to(OUT)), generation_script=str(Path(__file__).resolve()), generation_time=NOW, core_deliverable=core, usage=usage, alias_of=alias_of, output_sha256=sha(path)))
    CAPTIONS.append(dict(figure_id=figure_id, base=base, title=title, caption=caption, usage=usage, core=core, alias_of=alias_of))
    print('已生成 ' + base, flush=True)


def make_trends():
    all_data = []
    suffixes = ['Ritual_Court_Praise', 'Life_Career_Aging', 'Temple_Reclusion']
    for i, topic in enumerate(TOPICS):
        data = period_data(topic)
        all_data.append(data)
        fig, ax = plt.subplots(figsize=(7.8, 5.4))
        fig.subplots_adjust(left=.13, right=.95, bottom=.28, top=.78)
        title = f'{tid(topic)} {NAMES[topic]}：\n按作者所属时期划分的主题概率变化'
        fig.text(.10, .955, title, fontsize=14, va='top', linespacing=1.45)
        plot_period(ax, data)
        fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.53, .075), ncol=2)
        contrast = '初唐—晚唐、中唐—晚唐：跨模型稳健' if i == 0 else '初唐—晚唐：跨模型稳健'
        ax.text(.98, .96, contrast, ha='right', va='top', transform=ax.transAxes, fontsize=9, color='#555555')
        foot(fig, 'C_K20 · CORE · 95%作者簇 bootstrap CI')
        base = f'Figure_01{chr(97+i)}_{tid(topic)}'
        save(fig, base, f'1{chr(97+i)}', title.replace('\n', ''), data, trend_caption(topic), '独立权威图；正文图1组成面板', sources=('bcd_report', 'ladder'))
        save(fig, f'Figure_{i+2:02d}_{tid(topic)}_{suffixes[i]}_Trend', str(i+2), title.replace('\n', ''), data, trend_caption(topic), '补充材料或单题讨论；与图1面板择一', core=True, sources=('bcd_report', 'ladder'), alias_of=base)
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 9.3))
    fig.subplots_adjust(left=.14, right=.95, bottom=.14, top=.84, hspace=.49)
    heading(fig, '核心主题趋势总览', 'C_K20 · CORE · 按作者所属时期分组 · 95%作者簇 bootstrap CI')
    for i, (ax, topic, data) in enumerate(zip(axes, TOPICS, all_data)):
        plot_period(ax, data, compact=i < 2)
        ax.set_title(f'{chr(97+i)}  {tid(topic)}  {NAMES[topic]}', loc='left', pad=9)
        ax.set_ylabel('平均 Topic probability', fontsize=9.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.54, .043), ncol=2)
    foot(fig, '初唐—盛唐—中唐—晚唐为类别顺序；各面板纵轴均从0起，上限不同。')
    data = pd.concat(all_data, ignore_index=True)
    caption = '本图为图1a—1c的论文排版合图；独立面板为权威可视化。' + ' '.join(trend_caption(t) for t in TOPICS)
    base = 'Figure_01_Core_Robust_Topic_Trends'
    save(fig, base, '1', '核心主题趋势总览：按作者所属时期分组', data, caption, '正文优先', core=True, sources=('bcd_report', 'ladder'))
    save(fig, base + '_combined', '1-combined', '核心主题趋势总览：按作者所属时期分组', data, caption, '图1排版副本', sources=('bcd_report', 'ladder'), alias_of=base)
    plt.close(fig)


def model_legend(fig, y):
    handles = [Line2D([0], [0], color=v['color'], marker=v['marker'], linestyle=v['ls'], markersize=5, label=v['label']) for v in MODEL.values()]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, y), ncol=3)


def plot_effects(ax, data, findings, ylimits):
    for offset, (model, style) in zip((-.20, 0, .20), MODEL.items()):
        rows = data.loc[data.model.eq(model)].set_index('finding_id').loc[findings]
        y = rows.raw_delta.to_numpy()
        ax.errorbar(np.arange(len(findings)) + offset, y, yerr=[y-rows.bootstrap_ci_low, rows.bootstrap_ci_high-y], fmt=style['marker'], color=style['color'], markersize=5.5, capsize=3.5, elinewidth=1.1, markerfacecolor='white' if model != 'C' else style['color'], zorder=3)
    labels = []
    for fid in findings:
        topic = 'topic_' + fid[1:3]
        contrast = {'EL': '初唐—晚唐', 'ML': '中唐—晚唐'}[fid[-2:]]
        short = {'topic_03': '朝廷礼乐、\n应制与颂圣', 'topic_06': '老病身世、仕宦\n与人生感怀', 'topic_08': '山居寺院、\n僧侣与幽居'}[topic]
        labels.append(f'{tid(topic)}  {contrast}\n{short}')
    ax.set_xticks(range(len(findings)), labels, fontsize=9)
    ax.set_xlim(-.55, len(findings)-.45)
    ax.set_ylim(*ylimits)
    axis_style(ax)


def make_effects():
    all_data = effects(CORE)
    ylimits = limits(all_data)
    caption = '四组最终ROBUST_ACROSS_BCD finding的三模型效应比较；每组B/C/D点使用相同纵轴，两个聚合版本也使用相同尺度。' + EFFECT_NOTE + BCD_NOTE + TIME_NOTE + ''.join(mapping_caption(t) for t in TOPICS)
    for i, agg in enumerate(AGG):
        data = all_data.loc[all_data.aggregation.eq(agg)].copy()
        fig, ax = plt.subplots(figsize=(8.2, 5.7))
        fig.subplots_adjust(left=.13, right=.96, bottom=.27, top=.80)
        heading(fig, '四组核心发现的 B/C/D 效应比较', AGG[agg]['label'] + ' · CORE · 按作者所属时期分组')
        plot_effects(ax, data, CORE, ylimits)
        model_legend(fig, .10)
        foot(fig, '点为 raw Δ；误差线为95% CI；四组发现均为跨模型稳健。')
        save(fig, f'Figure_05{chr(97+i)}_BCD_Effects_{agg}', f'5{chr(97+i)}', 'B/C/D核心效应：' + AGG[agg]['label'] + '，按作者所属时期分组', data, caption, '正文优先，与另一聚合版本配对', core=True, sources=('bcd_report', 'mapping', 'ladder'))
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.1))
    fig.subplots_adjust(left=.13, right=.96, bottom=.15, top=.83, hspace=.61)
    heading(fig, '四组核心发现的 B/C/D 效应比较', '按作者所属时期分组；两种聚合分面、共享尺度')
    for ax, agg in zip(axes, AGG):
        plot_effects(ax, all_data.loc[all_data.aggregation.eq(agg)], CORE, ylimits)
        ax.set_title(AGG[agg]['label'], loc='left', pad=8)
    model_legend(fig, .035)
    foot(fig, '排版合图；独立版本见图5a与图5b。')
    save(fig, 'Figure_05_BCD_Core_Finding_Effect_Comparison', '5', 'B/C/D核心效应比较：按作者所属时期分组', all_data, caption, '图5a、5b的分面排版合图', sources=('bcd_report', 'mapping', 'ladder'))
    plt.close(fig)


def make_profiles():
    for i, topic in enumerate(TOPICS):
        rows = effects([tid(topic) + '_EL'])
        fields = ['mean_' + code for code in CODES]
        data = rows.melt(id_vars=[c for c in rows.columns if c not in fields], value_vars=fields, var_name='source_mean_column', value_name='mean_topic_probability')
        data['period'] = data.source_mean_column.map(dict(zip(fields, PERIODS)))
        data['period_order'] = data.period.map({p: i+1 for i, p in enumerate(PERIODS)})
        for agg in AGG:
            part = data.loc[data.aggregation.eq(agg)].sort_values(['model', 'period_order'])
            fig, ax = plt.subplots(figsize=(7.8, 5.1))
            fig.subplots_adjust(left=.13, right=.96, bottom=.30, top=.79)
            heading(fig, f'{tid(topic)} {NAMES[topic]}', AGG[agg]['label'] + ' · CORE · B/C/D · 按作者所属时期分组')
            for model, style in MODEL.items():
                line = part.loc[part.model.eq(model)].set_index('period').loc[PERIODS]
                ax.plot(range(4), line.mean_topic_probability, color=style['color'], linestyle=style['ls'], marker=style['marker'], markersize=5.5, linewidth=1.6, markerfacecolor='white' if model != 'C' else style['color'])
            axis_style(ax, True)
            ax.set_ylim(0, data.mean_topic_probability.max() * 1.18)
            model_legend(fig, .10)
            foot(fig, MAPPING[topic][3] + ('  ·  B/D为拆分主题概率和' if i < 2 else '  ·  一对一直接对应'))
            caption = f'{tid(topic)}“{NAMES[topic]}”在B/C/D CORE样本中按作者所属时期分组的四期概率profile，采用{AGG[agg]["label"]}。四个真实均值直接连接，不插值或平滑。均值直接转录最终效应表mean_E/H/M/L字段。该冻结表未保存B/D时期均值CI，本图仅展示点估计，不以效应CI冒充均值CI，也未新算区间。' + mapping_caption(topic) + BCD_NOTE + TIME_NOTE + '纵轴从0起；同一主题两聚合版本共享上限。具体对比的跨模型稳健不要求四期路径或峰值完全相同。'
            base = f'Figure_06{chr(97+i)}_{tid(topic)}_BCD_Period_Profile'
            save(fig, base + '_' + agg, f'6{chr(97+i)}-{agg}', f'{tid(topic)} B/C/D四期概率：按作者所属时期分组', part, caption, '正文候选' if agg == 'AUTHOR' else '补充材料', sources=('bcd_report', 'mapping', 'ladder'))
            if agg == 'AUTHOR':
                save(fig, base, f'6{chr(97+i)}', f'{tid(topic)} B/C/D四期概率：按作者所属时期分组', part, caption, '正文候选；默认AUTHOR版', core=True, sources=('bcd_report', 'mapping', 'ladder'), alias_of=base + '_AUTHOR')
            plt.close(fig)


def make_ladder():
    data = TABLE['ladder'].copy()
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    fig.subplots_adjust(left=.05, right=.97, bottom=.10, top=.82)
    heading(fig, '九组发现的跨模型稳健性阶梯', '按作者所属时期分组 · 顺序完全沿用最终阶梯')
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(9.7, -.6)
    for x, label in zip([.01, .10, .57, .77], ['顺序', '主题发现', '时期对比', '最终等级']):
        ax.text(x, -.45, label, fontsize=10, color='#666666', va='center')
    for i, row in enumerate(data.itertuples()):
        robust = row.finding_level_robustness == 'ROBUST_ACROSS_BCD'
        fill, color = ('#EEF5F2', '#236A53') if robust else ('#F6F3EB', '#7D6328')
        ax.add_patch(plt.Rectangle((0, i), 1, .86, color=fill, zorder=0))
        ax.text(.02, i+.43, str(row.robustness_rank), va='center', fontsize=10)
        ax.text(.10, i+.43, tid(row.topic) + '  ' + row.topic_label, va='center', fontsize=10)
        ct = '—'.join(PERIODS[CODES.index(c)] for c in row.contrast.split('-'))
        ax.text(.57, i+.43, ct, va='center', fontsize=10)
        ax.text(.77, i+.43, '跨模型稳健' if robust else '部分稳健', color=color, va='center', fontsize=11)
    foot(fig, '等级针对具体时期对比；排序不表示效应大小，也不表示文学史重要性。')
    caption = '九组finding逐行沿用finding_robustness_ladder.csv的顺序：T08 E-L、T03 E-L、T03 M-L、T06 E-L为跨模型稳健（ROBUST_ACROSS_BCD）；T16 E-L、T03 E-H、T11 M-L、T07 E-L、T19 E-H为部分稳健（PARTIALLY_ROBUST）。图中不将部分稳健表示为失败，也不赋予等级间数值距离。' + TIME_NOTE + BCD_NOTE
    save(fig, 'Figure_07_Finding_Robustness_Ladder', '7', '发现稳健性阶梯：按作者所属时期分组', data, caption, '正文优先', core=True, sources=('bcd_report',))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.8, 7.7))
    fig.subplots_adjust(left=.055, right=.97, bottom=.13, top=.82)
    heading(fig, '九组发现的证据矩阵', '按作者所属时期分组 · 原C统计与作者影响状态保持冻结值')
    ax.set_axis_off()
    columns = ['发现', 'C统计\n验证', '作者\n影响', 'B映射', 'D映射', '效应证据\n简记', '最终\n稳健性']
    words = {'DIRECT': '直接', 'BUNDLE': '组合', 'MERGED_PROXY': '合并代理'}
    notes = {'T08_EL': '同向且CI支持', 'T03_EL': '同向且CI支持', 'T03_ML': '同向且CI支持', 'T06_EL': '同向且CI支持', 'T16_EL': 'B区间跨0', 'T03_EH': 'B区间跨0', 'T11_ML': 'B/D区间跨0', 'T07_EL': '聚合/组合敏感', 'T19_EH': 'D为合并代理'}
    display = []
    for row in data.itertuples():
        display.append([row.finding_id.replace('_', ' '), '强支持' if row.C_original_status == 'STRONG' else '支持', '稳健' if row.C_author_robustness_status == 'AUTHOR_ROBUST' else '集中但\n稳健', words[row.B_mapping_type] + '\n' + ('高置信' if row.B_semantic_confidence == 'HIGH' else '中置信'), words[row.D_mapping_type] + '\n' + ('高置信' if row.D_semantic_confidence == 'HIGH' else '中置信'), notes[row.finding_id], '跨模型\n稳健' if row.finding_level_robustness == 'ROBUST_ACROSS_BCD' else '部分\n稳健'])
    table = ax.table(cellText=display, colLabels=columns, cellLoc='center', colWidths=[.115, .105, .12, .13, .13, .255, .12], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#DADADA')
        cell.set_linewidth(.5)
        cell.set_facecolor('#EAEDEA' if r == 0 else ('#F2F7F4' if r <= 4 else '#F9F7F1'))
    foot(fig, 'E/H/M/L＝初/盛/中/晚唐；映射置信度不等于新获人工批准。\n作者影响为C模型原有诊断；不能据此补称B/D完成完整LOAO。')
    save(fig, 'Figure_07b_Finding_Evidence_Matrix', '7b', '发现证据矩阵：按作者所属时期分组', data, caption + 'C统计强支持/支持对应STRONG/SUPPORTED；作者集中但稳健对应CONCENTRATED_BUT_ROBUST，稳健对应AUTHOR_ROBUST。效应证据简记只是冻结诊断的显示简称，不重做分类。T03 M-L未纳入原B/D额外Top1/Top3检查。', '补充材料', sources=('bcd_report', 'mapping', 'candidates'))
    plt.close(fig)


def make_t16():
    data = effects(['T16_EL'])
    ylimits = limits(data)
    fig, axes = plt.subplots(1, 2, figsize=(8.1, 5.6), sharey=True)
    fig.subplots_adjust(left=.13, right=.96, bottom=.23, top=.75, wspace=.17)
    heading(fig, 'T16 边塞征战与军旅：部分稳健', '初唐—晚唐 · CORE · 按作者所属时期分组 · 95%作者簇 bootstrap CI')
    for ax, agg in zip(axes, AGG):
        part = data.loc[data.aggregation.eq(agg)].set_index('model')
        for i, (model, style) in enumerate(MODEL.items()):
            row = part.loc[model]
            ax.errorbar(i, row.raw_delta, yerr=[[row.raw_delta-row.bootstrap_ci_low], [row.bootstrap_ci_high-row.raw_delta]], fmt=style['marker'], color=style['color'], markerfacecolor='white' if model != 'C' else style['color'], capsize=4, elinewidth=1.2, markersize=6)
        ax.set_xticks(range(3), ['B', 'C', 'D'])
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(*ylimits)
        axis_style(ax)
        ax.set_title(AGG[agg]['label'], pad=13, fontsize=11)
        ax.annotate('B：CI跨0', xy=(0, part.loc['B'].bootstrap_ci_high), xytext=(.25, ylimits[1]-.01), fontsize=9, arrowprops={'arrowstyle': '-', 'color': '#777777', 'lw': .7})
    axes[1].set_ylabel('')
    model_legend(fig, .08)
    foot(fig, 'B/C/D差值方向均为负；B的两种区间均跨0，故保留为部分稳健。')
    caption = 'T16“边塞征战与军旅”初唐—晚唐效应的两种聚合分面。B/C/D六个点估计方向均为负，B的POEM与AUTHOR区间均跨0，C/D对应区间均位于零线下方；部分稳健（PARTIALLY_ROBUST）不等于方向相反。B的不确定性相对于其较小点估计足以覆盖零，不应概括为B的区间绝对宽度在两种聚合中都大于C/D。' + EFFECT_NOTE + BCD_NOTE + TIME_NOTE + mapping_caption('topic_16')
    save(fig, 'Figure_08_T16_Border_War_Partial_Robustness', '8', 'T16部分稳健案例：按作者所属时期分组', data, caption, '补充材料；方法讨论可选正文', core=True, sources=('bcd_report', 'ladder', 'mapping'))
    plt.close(fig)


def make_scatter():
    rows = effects(['T03_EL', 'T06_EL', 'T08_EL', 'T16_EL'], model='C')
    data = rows.pivot(index=['finding_id', 'c_topic_id', 'c_topic_label', 'contrast', 'model', 'sample'], columns='aggregation', values=['raw_delta', 'source_row']).reset_index()
    data.columns = ['_'.join(c).rstrip('_') if isinstance(c, tuple) else c for c in data.columns]
    data['source_file'] = str(SOURCE['effects'])
    data['finding_level_robustness'] = data.finding_id.map(TABLE['ladder'].set_index('finding_id').finding_level_robustness)
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    fig.subplots_adjust(left=.16, right=.95, bottom=.19, top=.81)
    heading(fig, '两种聚合口径的效应对照', 'C_K20 · CORE · 初唐—晚唐 · 按作者所属时期分组')
    bounds = (-.165, .135)
    ax.set_xlim(*bounds)
    ax.set_ylim(*bounds)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='#888888', lw=.8)
    ax.axvline(0, color='#888888', lw=.8)
    ax.plot(bounds, bounds, color='#A0A0A0', linestyle='--', lw=.9, zorder=0)
    ax.text(.109, .126, 'Δ相等', fontsize=9, color='#777777', ha='right')
    positions = {'T03_EL': (-.148, -.065), 'T06_EL': (.008, .121), 'T08_EL': (.008, .033), 'T16_EL': (-.148, -.005)}
    annotations = []
    for row in data.itertuples():
        robust = row.finding_level_robustness == 'ROBUST_ACROSS_BCD'
        ax.scatter(row.raw_delta_POEM, row.raw_delta_AUTHOR, marker='o' if robust else 's', s=58, edgecolor='#333333', facecolor='#555555' if robust else 'white', zorder=3)
        label = tid(row.c_topic_id) + ' ' + row.c_topic_label
        if row.c_topic_id == 'topic_03':
            label = label.replace('应制', '\n应制')
        elif row.c_topic_id == 'topic_06':
            label = label.replace('与人生', '\n与人生')
        elif row.c_topic_id == 'topic_08':
            label = label.replace('僧侣', '\n僧侣')
        if not robust:
            label += '\n（部分稳健）'
        annotations.append(ax.annotate(label, (row.raw_delta_POEM, row.raw_delta_AUTHOR), xytext=positions[row.finding_id], textcoords='data', va='top', fontsize=9, arrowprops={'arrowstyle': '-', 'color': '#BBBBBB', 'lw': .6}, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5}, linespacing=1.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set_xlabel('POEM Δ（按诗等权）', labelpad=10)
    ax.set_ylabel('AUTHOR Δ（作者均衡）', labelpad=10)
    # 注释底色不得遮住任何数据点，逐点检查显示坐标。
    fig.canvas.draw()
    for number, annotation in enumerate(annotations):
        box = annotation.get_bbox_patch().get_window_extent().padded(5)
        points = ax.transData.transform(data[['raw_delta_POEM', 'raw_delta_AUTHOR']].to_numpy())
        check(f'图9注释{number+1}不遮数据点', not any(box.contains(x, y) for x, y in points))
    foot(fig, '同一象限表示方向一致；靠近45度虚线表示两种聚合的效应大小相近。')
    caption = 'C_K20 CORE中T03、T06、T08、T16四组E-L finding的点估计对照，横轴为POEM Δ，纵轴为AUTHOR Δ，均直接取自最终BCD效应表。两轴使用相同范围和等比例显示；虚线为AUTHOR Δ=POEM Δ，横纵零线分隔方向。T16用空心方点及文字标记其PARTIALLY_ROBUST等级，其余三项为ROBUST_ACROSS_BCD。此图仅比较两个estimand的方向与大小，不作相关检验、显著性检验或新分类。' + TIME_NOTE
    save(fig, 'Figure_09_POEM_vs_AUTHOR_Core_Findings', '9', 'POEM与AUTHOR效应对照：按作者所属时期分组', data, caption, '补充材料', core=True, sources=('bcd_report', 'ladder'))
    plt.close(fig)


def exclusion_data():
    source = TABLE['exclusion']
    rows = source.loc[source.topic_id.isin(TOPICS) & source.contrast.eq('E-L') & source.exclusion_scheme.eq('GLOBAL_TOP') & source.N.isin([1, 3, 5])].copy()
    keys = ['sample', 'topic_id', 'contrast', 'aggregation', 'exclusion_scheme', 'N']
    ci = TABLE['exclusion_ci'][keys + ['delta', 'ci_low', 'ci_high', 'bootstrap_iterations', 'seed', 'source_file', 'source_row']].rename(columns={'source_file': 'ci_source_file', 'source_row': 'ci_source_row'})
    rows = rows.merge(ci, on=keys, validate='one_to_one')
    close('作者排除点估计与CI来源', rows.new_delta, rows.delta, 0)
    original = []
    for topic in TOPICS:
        for agg in AGG:
            selected = rows.loc[rows.topic_id.eq(topic) & rows.aggregation.eq(agg)]
            check(f'{topic}/{agg} Top1/3/5完整且无反向', len(selected) == 3 and not selected.sign_flip.any())
            primary = TABLE['primary'].loc[lambda x: x.topic_id.eq(topic) & x.contrast.eq('E-L')].iloc[0]
            close(f'{topic}/{agg} Original核对', selected.original_delta, primary['raw_effect_' + agg.lower()])
            original.append(dict(sample='CORE', topic_id=topic, topic_label=NAMES[topic], contrast='E-L', aggregation=agg, exclusion_scheme='GLOBAL_TOP', N=0, new_delta=selected.original_delta.iloc[0], ci_low=primary[agg.lower() + '_ci_low'], ci_high=primary[agg.lower() + '_ci_high'], bootstrap_iterations=5000, seed=20260901, source_file=selected.source_file.iloc[0], source_row=selected.source_row.iloc[0], ci_source_file=str(SOURCE['primary']), ci_source_row=primary.source_row, excluded_author_count=0))
    data = pd.concat([pd.DataFrame(original), rows], ignore_index=True)
    data['model'] = 'C'
    data['stage'] = data.N.map({0: '原始', 1: '剔除Top1', 3: '剔除Top3', 5: '剔除Top5'})
    return data


def plot_exclusion(ax, data, ylimits):
    for topic, style in TOPIC_STYLE.items():
        rows = data.loc[data.topic_id.eq(topic)].set_index('N').loc[[0, 1, 3, 5]]
        y = rows.new_delta.to_numpy()
        ax.errorbar(range(4), y, yerr=[y-rows.ci_low, rows.ci_high-y], color=style['color'], linestyle=style['ls'], marker=style['marker'], markersize=5, linewidth=1.4, elinewidth=.8, capsize=3, label=tid(topic) + ' ' + NAMES[topic])
    axis_style(ax)
    ax.set_xticks(range(4), ['原始', '剔除Top1', '剔除Top3', '剔除Top5'])
    ax.set_ylim(*ylimits)
    ax.set_xlim(-.22, 3.22)


def make_exclusion():
    data = exclusion_data()
    ylimits = limits(data, 'ci_low', 'ci_high')
    caption = 'C_K20 CORE的T03、T06、T08初唐—晚唐finding在原始及GLOBAL_TOP Top1/Top3/Top5排除后的效应。Top作者按冻结流水线的Topic贡献排名选择，删除单位为作者整簇；本图沿用GLOBAL_TOP主方案，不以PER_PERIOD_TOP替换。连线仅表示四种离散排除情景。原始点取排除表original_delta，原始CI取此前效应验证表（5,000次，seed=20260901）；排除后点及CI取原作者影响表（2,000次，基准seed=20260902、各情景采用冻结派生种子）。各区间保留所属阶段来源，不将原始CI继承给删除情景。POEM与AUTHOR分面并共享尺度；方向保持不等于每个排除后CI均不跨零，也不等于效应大小不变。' + TIME_NOTE
    for i, agg in enumerate(AGG):
        part = data.loc[data.aggregation.eq(agg)].copy()
        fig, ax = plt.subplots(figsize=(8.1, 5.9))
        fig.subplots_adjust(left=.13, right=.96, bottom=.31, top=.78)
        heading(fig, '核心发现的作者排除敏感性', AGG[agg]['label'] + ' · C_K20 CORE · 按作者所属时期分组 · E-L')
        plot_exclusion(ax, part, ylimits)
        fig.legend(*ax.get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.15, .065), ncol=1, fontsize=9)
        foot(fig, 'GLOBAL_TOP：按主题贡献排除作者；误差线为各情景已有95% CI。')
        save(fig, f'Figure_10{chr(97+i)}_Author_Influence_{agg}', f'10{chr(97+i)}', '作者排除敏感性：' + AGG[agg]['label'] + '，按作者所属时期分组', part, caption, '补充材料', sources=('influence_report', 'primary'))
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8.1, 9.0))
    fig.subplots_adjust(left=.13, right=.96, bottom=.22, top=.83, hspace=.45)
    heading(fig, '核心发现的作者排除敏感性', 'C_K20 CORE · 按作者所属时期分组 · 初唐—晚唐')
    for ax, agg in zip(axes, AGG):
        plot_exclusion(ax, data.loc[data.aggregation.eq(agg)], ylimits)
        ax.set_title(AGG[agg]['label'], loc='left', pad=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower left', bbox_to_anchor=(.15, .055), ncol=1, fontsize=9)
    foot(fig, 'GLOBAL_TOP；两个聚合分面共享尺度，区间来自各冻结阶段。')
    save(fig, 'Figure_10_Author_Influence_Core_Findings', '10', '核心发现作者排除敏感性：按作者所属时期分组', data, caption, '补充材料；方法讨论可选正文', core=True, sources=('influence_report', 'primary'))
    plt.close(fig)


def write_documentation(font, candidates):
    pd.DataFrame(MANIFEST).to_csv(OUT / 'figure_manifest.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(LAYOUT).to_csv(OUT / 'qa/layout_bounds.csv', index=False, encoding='utf-8-sig')
    source_rows = []
    for path, value in HASHES.items():
        after = sha(path)
        source_rows.append(dict(path=path, sha256_before=value, sha256_after=after, unchanged=value == after))
    check('全部冻结来源前后哈希不变', all(r['unchanged'] for r in source_rows))
    pd.DataFrame(source_rows).to_csv(OUT / 'qa/source_hashes.csv', index=False, encoding='utf-8-sig')
    lines = ['# 论文图题与图注', '', '本文件仅描述已冻结结果；按作者所属时期分组，不新增文学史解释。', '']
    for item in CAPTIONS:
        lines += [f"## Figure {item['figure_id']}：{item['title']}", '', f"文件：`{item['base']}.png/.pdf/.svg`", '', item['caption'], '', '建议用途：' + item['usage'] + '。', '']
    (OUT / 'publication_figure_captions.md').write_text('\n'.join(lines), encoding='utf-8')
    readme = ['# 《全唐诗》作者时期主题趋势论文图', '', '已生成10个图组、13个最低要求核心图位；共27个文件版本，每版均有600 dpi PNG、矢量PDF、SVG和对应plot_data CSV。重复/排版版本不代表新增发现。', '', '## 来源与复现', '', '脚本只读取既有结果CSV，不读取概率矩阵计算均值，不重新bootstrap、训练、分类、推断、平滑或更改映射。同名候选中选择最新已完成且manifest QC通过的目录；使用前验证冻结哈希。', '', '```powershell', 'python historical_analysis/publication_figures/scripts/generate_publication_figures.py', '```', '', '仅核查来源和数值：命令末尾加 `--audit-only`。数值或映射核对不一致时脚本中止。', '', '实际字体：' + font.get('source_font_family', font['font_family']) + '（常规400字重）' + '；字体文件：`' + font['font_file'] + '`。PDF嵌入TrueType字体，SVG转路径。', '', '## 使用说明', '', '- 正文优先：图1、图5a/5b、图7；需要跨模型四期路径时选图6的AUTHOR版本。', '- 补充材料：图2—4（正文已有图1时）、图6的POEM版、图7b、图8—10。图8或图10可用于正文方法讨论。', '- 图1a—1c为独立权威图；图1及combined为排版合图。图2—4与其相应面板内容一致。', '- 图6无后缀文件与AUTHOR版内容一致；图5和图10另有分面合图。', '- 概率纵轴从0起；跨模型效应同图共轴，图5两聚合版本同尺度，图9两轴等比例。', '- 两聚合并画时：POEM蓝实线实心圆，AUTHOR棕虚线空心方。三模型并画时：B蓝虚线三角、C绿实线圆、D紫点线菱形；聚合由标题/分面明确。图10用固定主题编码及聚合分面。黑白打印仍可区分。', '- 每图来源、行数、生成时间和脚本见figure_manifest.csv；完整精度与原CSV行号见data目录。', '', '## 需要保留的说明', '', '- 图1—4均值CI来自趋势阶段2,000次bootstrap；图5/8效应CI来自BCD阶段5,000次bootstrap，不同对象及阶段未混用。', '- 图6直接使用BCD表四期点估计；该表未保存B/D四期均值CI，所以图6不画CI，不新增计算。', '- T03并非持续下降：盛唐至中唐有回升。T08的B/D AUTHOR路径也非持续上升；保留所有真实均值。', '- T16六个点估计同向为负，B两种CI跨0；B区间不一定在绝对宽度上比C/D更宽。', '- T03/T06的B/D是split主题概率和；T08是HIGH DIRECT。来源未记录正式human_review_status，不补称人工确认。', '- 图10保留原GLOBAL_TOP方案及实际删除名单，不混同BCD额外检查的GLOBAL_TOP_WITHIN_CONTRAST。', '- 原trend_shape数据字典释义不一致已在项目记忆记录，本任务不改冻结字典，图只读真实均值。', '', '## 文件索引', '', '| 文件基名 | 用途 |', '|---|---|']
    readme += [f"| [{c['base']}](png/{c['base']}.png) | {c['usage']} |" for c in CAPTIONS]
    readme += ['', '图注：publication_figure_captions.md；数值和来源核验：qa目录。PDF渲染与逐图版面检查由独立验收脚本及人工检查记录。', '', '下一步仅由用户选择正文图和排版位置；本任务完成后停止。']
    (OUT / 'README.md').write_text('\n'.join(readme) + '\n', encoding='utf-8')
    payload = dict(generation_time=NOW, python=sys.executable, python_version=sys.version, matplotlib=matplotlib.__version__, pandas=pd.__version__, font=font, selected_sources={k: str(v) for k, v in SOURCE.items()}, source_candidates=candidates, numeric_checks=len(AUDIT), all_numeric_checks_passed=all(c['status'] == 'PASS' for c in AUDIT), figure_versions=len(CAPTIONS), core_slots=sum(c['core'] for c in CAPTIONS), formats=['png', 'pdf', 'svg'], png_dpi=600, recomputed_statistics=False, generation_script_sha256=sha(__file__))
    (OUT / 'generation_metadata.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    for folder in ('data', 'png', 'pdf', 'svg', 'qa'):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    try:
        candidates = locate_sources()
        preflight()
        print(f'来源和数值预检通过：{len(AUDIT)}项；{len(HASHES)}个冻结文件', flush=True)
        if args.audit_only:
            return
        font = configure_font()
        make_trends()
        make_effects()
        make_profiles()
        make_ladder()
        make_t16()
        make_scatter()
        make_exclusion()
        check('27个文件版本', len(CAPTIONS) == 27)
        check('13个核心图位', sum(c['core'] for c in CAPTIONS) == 13)
        write_documentation(font, candidates)
        print('绘图完成；继续独立PDF/PNG与plot_data核验。', flush=True)
    finally:
        pd.DataFrame(AUDIT).to_csv(OUT / 'qa/numeric_and_source_checks.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
