"""独立回读逐图底表、PDF和PNG，并渲染全部PDF供人工验收。"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys

import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from pypdf import PdfReader

OUT = Path(__file__).resolve().parents[1]
QA = OUT / 'qa'
RENDERS = QA / 'rendered'
CHECKS = []
SOURCES = {}


def check(name, passed, detail=''):
    CHECKS.append(dict(check=name, status='PASS' if passed else 'FAIL', detail=str(detail)))
    if not passed:
        raise AssertionError(f'{name}: {detail}')


def source(path, row):
    if path not in SOURCES:
        SOURCES[path] = pd.read_csv(path, float_precision='round_trip', encoding='utf-8-sig')
    return SOURCES[path].iloc[int(row)-2]


def compare_values(label, row, expected, fields):
    for actual_col, source_col in fields:
        actual, value = row[actual_col], expected[source_col]
        if isinstance(value, str):
            check(label + '/' + actual_col, actual == value)
        else:
            check(label + '/' + actual_col, float(actual) == float(value), f'{actual!r} vs {value!r}')


def verify_data(path):
    data = pd.read_csv(path, float_precision='round_trip', encoding='utf-8-sig')
    name = path.stem
    for i, row in data.iterrows():
        key = f'{name}/{i+2}'
        if name.startswith('Figure_09'):
            for agg in ('POEM', 'AUTHOR'):
                original = source(row.source_file, row['source_row_' + agg])
                compare_values(key + '/' + agg, row, original, [('raw_delta_' + agg, 'raw_delta'), ('finding_id', 'finding_id'), ('model', 'model')])
                check(key + '/聚合', original.aggregation == agg)
            continue
        original = source(row.source_file, row.source_row)
        if name.startswith('Figure_06'):
            compare_values(key, row, original, [('mean_topic_probability', row.source_mean_column), ('model', 'model'), ('aggregation', 'aggregation'), ('mapped_topic_ids', 'mapped_topic_ids')])
            check(key + '/时期', row.period == dict(mean_E='初唐', mean_H='盛唐', mean_M='中唐', mean_L='晚唐')[row.source_mean_column])
        elif name.startswith('Figure_10'):
            compare_values(key, row, original, [('new_delta', 'original_delta' if row.N == 0 else 'new_delta'), ('topic_id', 'topic_id'), ('aggregation', 'aggregation')])
            ci = source(row.ci_source_file, row.ci_source_row)
            fields = [('ci_low', row.aggregation.lower() + '_ci_low'), ('ci_high', row.aggregation.lower() + '_ci_high')] if row.N == 0 else [('ci_low', 'ci_low'), ('ci_high', 'ci_high'), ('new_delta', 'delta'), ('N', 'N')]
            compare_values(key + '/CI', row, ci, fields)
        elif name.startswith(('Figure_05', 'Figure_08')):
            compare_values(key, row, original, [(c, c) for c in ['raw_delta', 'bootstrap_ci_low', 'bootstrap_ci_high', 'model', 'aggregation', 'finding_id', 'mapped_topic_ids']])
        elif name.startswith('Figure_07'):
            compare_values(key, row, original, [(c, c) for c in ['finding_id', 'robustness_rank', 'finding_level_robustness', 'B_mapping_type', 'D_mapping_type', 'C_author_robustness_status', 'C_original_status']])
        else:
            compare_values(key, row, original, [(c, c) for c in ['mean_topic_probability', 'topic_id', 'period', 'aggregation']])
            ci = source(row.ci_source_file, row.ci_source_row)
            compare_values(key + '/CI', row, ci, [('ci_2_5', 'ci_2_5'), ('ci_97_5', 'ci_97_5')])
    return len(data)


def inspect_resources(resources):
    images, embedded_fonts = 0, 0
    resources = resources.get_object()
    for ref in resources.get('/XObject', {}).get_object().values() if '/XObject' in resources else []:
        obj = ref.get_object()
        if obj.get('/Subtype') == '/Image':
            images += 1
        elif obj.get('/Subtype') == '/Form' and '/Resources' in obj:
            n, f = inspect_resources(obj['/Resources'])
            images, embedded_fonts = images+n, embedded_fonts+f
    if '/Font' in resources:
        for ref in resources['/Font'].get_object().values():
            font = ref.get_object()
            family = font.get('/DescendantFonts', [font])
            for child in family:
                child = child.get_object()
                if '/FontDescriptor' in child:
                    descriptor = child['/FontDescriptor'].get_object()
                    embedded_fonts += any(key in descriptor for key in ('/FontFile', '/FontFile2', '/FontFile3'))
    return images, embedded_fonts


def contact_sheets(images):
    # 缩略审查页仅用于定位，最终仍逐张打开渲染图验收。
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 19)
    for start in range(0, len(images), 6):
        sheet = Image.new('RGB', (1800, 1680), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, path in enumerate(images[start:start+6]):
            img = Image.open(path).convert('RGB')
            img.thumbnail((870, 495))
            x, y = (n % 2)*900, (n // 2)*560
            sheet.paste(img, (x + (900-img.width)//2, y+40))
            draw.text((x+15, y+8), path.stem.replace('Figure_', 'F')[:66], font=font, fill='#333333')
        sheet.save(QA / f'contact_{start//6+1:02d}.png')


def main():
    RENDERS.mkdir(exist_ok=True)
    manifest = pd.read_csv(OUT / 'figure_manifest.csv', encoding='utf-8-sig')
    layout = pd.read_csv(QA / 'layout_bounds.csv').set_index('figure')
    check('81个格式化图文件', len(manifest) == 81)
    check('27个逐图底表', manifest.plot_data_file.nunique() == 27)
    for filename in manifest.plot_data_file.unique():
        count = verify_data(OUT / filename)
        check(filename + '/manifest行数', set(manifest.loc[manifest.plot_data_file.eq(filename), 'data_rows_used']) == {count})
    renders = []
    file_audit = []
    for row in manifest.itertuples():
        path = OUT / row.figure_filename
        check(path.name + '/输出哈希', hashlib.sha256(path.read_bytes()).hexdigest() == row.output_sha256)
        if path.suffix == '.png':
            with Image.open(path) as img:
                dpi = img.info.get('dpi', (0, 0))
                check(path.name + '/600dpi', all(abs(v-600) < .1 for v in dpi), dpi)
                expected = layout.loc[path.stem]
                size = (round(expected.width_in*600), round(expected.height_in*600))
                check(path.name + '/像素尺寸', img.size == size, img.size)
                img.verify()
        elif path.suffix == '.pdf':
            pdf = PdfReader(path, strict=True)
            check(path.name + '/单页可打开', len(pdf.pages) == 1)
            page = pdf.pages[0]
            images, fonts = inspect_resources(page['/Resources'])
            check(path.name + '/矢量非栅格PDF', images == 0, images)
            check(path.name + '/字体嵌入', fonts > 0, fonts)
            text = page.extract_text()
            check(path.name + '/作者时期中文文本', '按作者所属时期' in text, text[:100])
            check(path.name + '/无乱码替代符', '\ufffd' not in text)
            prefix = RENDERS / path.stem
            proc = subprocess.run(['pdftoppm', '-singlefile', '-png', '-scale-to', '1600', str(path), str(prefix)], capture_output=True, text=True, encoding='utf-8', errors='replace')
            check(path.name + '/Poppler渲染', proc.returncode == 0 and prefix.with_suffix('.png').exists(), proc.stderr[:300])
            renders.append(prefix.with_suffix('.png'))
            file_audit.append(dict(figure=path.stem, pdf_pages=1, embedded_font_count=fonts, raster_image_count=images, render_status='PASS'))
            print('已回读并渲染 ' + path.name, flush=True)
    hashes = pd.read_csv(QA / 'source_hashes.csv')
    for row in hashes.itertuples():
        check('冻结文件不变/' + Path(row.path).name, hashlib.sha256(Path(row.path).read_bytes()).hexdigest() == row.sha256_before)
    contact_sheets(renders)
    pd.DataFrame(file_audit).to_csv(QA / 'pdf_png_verification.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(CHECKS).to_csv(QA / 'independent_verification.csv', index=False, encoding='utf-8-sig')
    (QA / 'independent_verification_summary.json').write_text(json.dumps(dict(checks=len(CHECKS), all_passed=True, pdfs_rendered=len(renders), source_files_unchanged=len(hashes), verifier_python=sys.executable, manual_visual_review='PENDING'), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'独立核验通过：{len(CHECKS)}项；27张PDF已全部渲染。', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        pd.DataFrame(CHECKS).to_csv(QA / 'independent_verification.csv', index=False, encoding='utf-8-sig')
