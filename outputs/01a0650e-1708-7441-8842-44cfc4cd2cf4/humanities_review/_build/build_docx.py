"""按权威Markdown制作同文Word阅读版，使用中文字体及明确表格几何。"""
from pathlib import Path
import json
import re
import zipfile
from urllib.parse import quote
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Twips
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE as RT

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
text=(OUT/'QTS_BCD_Humanities_Review_Report.md').read_text(encoding='utf-8')
refs=dict(re.findall(r'^\[(S\w+)\]: <([^>]+)>$',text,re.M))
lines=[l for l in text.splitlines() if not re.match(r'^\[S\w+\]:',l)]
doc=Document()
sec=doc.sections[0]
sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=sec.left_margin=sec.right_margin=Inches(1)
sec.header_distance=sec.footer_distance=Inches(.492)

def style(name,size,before,after,color='222222',bold=False,line=1.333):
    s=doc.styles[name]
    s.font.name='Calibri'; s.font.size=Pt(size); s.font.bold=bold
    s.font.color.rgb=RGBColor.from_string(color)
    s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体' if name=='Normal' else '等线')
    p=s.paragraph_format; p.space_before=Pt(before); p.space_after=Pt(after)
    p.line_spacing=line; p.widow_control=True
    return s

style('Normal',11,0,8)
doc.styles['Normal'].paragraph_format.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
style('Title',24,0,16,'0B2545',True)
style('Subtitle',12,0,12,'586575')
for name,size,before,after,color in [('Heading 1',16,18,10,'2E74B5'),('Heading 2',13,12,6,'2E74B5'),('Heading 3',12,8,4,'1F4D78')]:
    style(name,size,before,after,color,True)
    doc.styles[name].paragraph_format.keep_with_next=True
for name in ['List Bullet','List Number']:
    style(name,11,0,4,line=1.208)
    doc.styles[name].paragraph_format.left_indent=Inches(.375)
    doc.styles[name].paragraph_format.first_line_indent=Inches(-.194)
for name in ['Source','Callout','TableText']:
    if name not in doc.styles:
        doc.styles.add_style(name,1)
style('Source',9,4,4,'586575',line=1.167)
style('Callout',11,6,8,'1F3A5F',line=1.333)
style('TableText',9,0,0,line=1.167)
doc.styles['Callout'].paragraph_format.left_indent=Inches(.15)
doc.styles['Callout'].paragraph_format.right_indent=Inches(.15)
style('Header',9,0,0,'586575',line=1)
style('Footer',9,0,0,'586575',line=1)
sec.header.paragraphs[0].text='《全唐诗》作者时期主题结构 · 人文学审读报告'
footer=sec.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run('第 ')
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); footer._p.append(fld)
footer.add_run(' 页')

# 显式设置真实列表编号，避免依赖不同Word环境的默认缩进。
num_root=doc.part.numbering_part.element
for lvl in num_root.findall('.//'+qn('w:lvl')):
    ppr=lvl.find(qn('w:pPr'))
    if ppr is None:
        ppr=OxmlElement('w:pPr'); lvl.append(ppr)
    ind=ppr.find(qn('w:ind'))
    if ind is None:
        ind=OxmlElement('w:ind'); ppr.append(ind)
    ind.set(qn('w:left'),'540'); ind.set(qn('w:hanging'),'279')
    tabs=OxmlElement('w:tabs'); tab=OxmlElement('w:tab')
    tab.set(qn('w:val'),'num'); tab.set(qn('w:pos'),'540'); tabs.append(tab); ppr.append(tabs)

expected=[]
def plain(s):
    return s.replace('**','').replace('`','')

def inline(p,s):
    expected.append(plain(s))
    for part in re.split(r'(\*\*[^*]+\*\*|`[^`]+`|\[S\w+\])',s):
        if not part:
            continue
        if part.startswith('[') and part[1:-1] in refs:
            href=refs[part[1:-1]]
            uri='file:///'+quote(href,safe='/:')
            rid=p.part.relate_to(uri,RT.HYPERLINK,is_external=True)
            hyp=OxmlElement('w:hyperlink'); hyp.set(qn('r:id'),rid)
            run=OxmlElement('w:r'); rp=OxmlElement('w:rPr')
            col=OxmlElement('w:color'); col.set(qn('w:val'),'2E74B5'); rp.append(col)
            run.append(rp); tt=OxmlElement('w:t'); tt.text=part; run.append(tt)
            hyp.append(run); p._p.append(hyp)
        else:
            run=p.add_run(plain(part))
            if part.startswith('**'):
                run.bold=True
            if part.startswith('`'):
                run.font.name='Consolas'; run.font.size=Pt(9)

def paragraph(s,name='Normal'):
    p=doc.add_paragraph(style=name)
    inline(p,s)
    if name=='Callout':
        sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'F4F6F9'); p._p.get_or_add_pPr().append(sh)
    return p

def add_table(block):
    cells=[[c.strip() for c in line.strip('|').split('|')] for line in block]
    cells=[cells[0]]+cells[2:]
    n=len(cells[0]); assert all(len(r)==n for r in cells)
    if n==6 and cells[0][0]=='Finding' and cells[0][1]=='模型':
        widths=[1170,470,1080,2780,1080,2780]
    elif n==6:
        widths=[1270,1120,1500,1480,2290,1700]
    elif n==5 and cells[0][0]=='模型' and 'mapping类型' in cells[0]:
        widths=[650,1200,1260,2970,3280]
    elif n==5 and cells[0][-1]=='需要注意':
        widths=[650,1150,1150,2180,4230]
    elif n==5:
        widths=[1650,1250,1250,1900,3310]
    elif n==4:
        widths=[1200,5440,1360,1360]
    else:
        widths=[1200,3100,5060]
    assert len(widths)==n and sum(widths)==9360
    table=doc.add_table(rows=0,cols=n)
    table.alignment=WD_TABLE_ALIGNMENT.LEFT; table.autofit=False
    for col,width in zip(table.columns,widths): col.width=Twips(width)
    pr=table._tbl.tblPr
    tw=pr.find(qn('w:tblW')); tw.set(qn('w:type'),'dxa'); tw.set(qn('w:w'),'9360')
    indent=OxmlElement('w:tblInd'); indent.set(qn('w:type'),'dxa'); indent.set(qn('w:w'),'120'); pr.append(indent)
    margins=OxmlElement('w:tblCellMar')
    for key,val in [('top',80),('bottom',80),('start',120),('end',120)]:
        elem=OxmlElement('w:'+key); elem.set(qn('w:type'),'dxa'); elem.set(qn('w:w'),str(val)); margins.append(elem)
    pr.append(margins)
    borders=OxmlElement('w:tblBorders')
    for key in ['top','left','bottom','right','insideH','insideV']:
        elem=OxmlElement('w:'+key); elem.set(qn('w:val'),'single'); elem.set(qn('w:sz'),'4'); elem.set(qn('w:color'),'D4DCE4'); borders.append(elem)
    pr.append(borders)
    for ri,row_values in enumerate(cells):
        row=table.add_row()
        no=OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(no)
        if ri==0:
            repeat=OxmlElement('w:tblHeader'); row._tr.get_or_add_trPr().append(repeat)
        for ci,(cell,value,width) in enumerate(zip(row.cells,row_values,widths)):
            cell.width=Twips(width); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p=cell.paragraphs[0]; p.style=doc.styles['TableText']
            p.paragraph_format.alignment=WD_ALIGN_PARAGRAPH.CENTER if ci<3 else WD_ALIGN_PARAGRAPH.LEFT
            inline(p,value)
            if ri==0:
                for run in p.runs: run.bold=True
                sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'F4F6F9'); cell._tc.get_or_add_tcPr().append(sh)
    return table

# 为两个独立编号清单分别从1开始，避免Word沿用前一节编号。
abstract_id=max(int(x.get(qn('w:abstractNumId'))) for x in num_root.findall(qn('w:abstractNum')))+1
abstract=OxmlElement('w:abstractNum'); abstract.set(qn('w:abstractNumId'),str(abstract_id))
lvl=OxmlElement('w:lvl'); lvl.set(qn('w:ilvl'),'0')
for tag,attrs in [('start',{'val':'1'}),('numFmt',{'val':'decimal'}),('lvlText',{'val':'%1.'}),('lvlJc',{'val':'left'})]:
    elem=OxmlElement('w:'+tag)
    for key,value in attrs.items(): elem.set(qn('w:'+key),value)
    lvl.append(elem)
ppr=OxmlElement('w:pPr'); ind=OxmlElement('w:ind'); ind.set(qn('w:left'),'540'); ind.set(qn('w:hanging'),'279'); ppr.append(ind)
tabs=OxmlElement('w:tabs'); tab=OxmlElement('w:tab'); tab.set(qn('w:val'),'num'); tab.set(qn('w:pos'),'540'); tabs.append(tab); ppr.append(tabs)
lvl.append(ppr); abstract.append(lvl); num_root.append(abstract)
current_num_id=None
list_numbers=[]
i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():
        i+=1; continue
    if line.startswith('|'):
        block=[]
        while i<len(lines) and lines[i].startswith('|'):
            block.append(lines[i]); i+=1
        add_table(block); continue
    match=re.match(r'^(#{1,6}) (.+)$',line)
    if match:
        level=len(match.group(1)); label=match.group(2)
        if level==1:
            p=paragraph(label,'Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        else:
            paragraph(label,f'Heading {min(level-1,3)}')
    elif line.startswith('**从LDA'):
        p=paragraph(line,'Subtitle'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    elif line.startswith('审读依据：'):
        paragraph(line,'Source')
        doc.add_page_break()
    elif line.startswith('> '):
        paragraph(line[2:],'Callout')
    elif line.startswith('- '):
        paragraph(line[2:],'List Bullet')
    elif re.match(r'^\d+\. ',line):
        ordinal=int(line.split('.')[0])
        if ordinal==1:
            current_num_id=max(int(x.get(qn('w:numId'))) for x in num_root.findall(qn('w:num')))+1
            num=OxmlElement('w:num'); num.set(qn('w:numId'),str(current_num_id))
            aid=OxmlElement('w:abstractNumId'); aid.set(qn('w:val'),str(abstract_id)); num.append(aid); num_root.append(num)
        p=paragraph(re.sub(r'^\d+\. ','',line),'List Number')
        np=p._p.get_or_add_pPr().get_or_add_numPr()
        np.get_or_add_ilvl().val=0; np.get_or_add_numId().val=current_num_id
        list_numbers.append((current_num_id,ordinal))
    elif line.startswith(('来源：','依据：')):
        paragraph(line,'Source')
    else:
        paragraph(line)
    i+=1
doc.core_properties.title='《全唐诗》作者时期主题结构：B–C–D跨模型稳健发现的人文学审读报告'
doc.core_properties.subject='从LDA统计结果到文学史解释之前的证据审查'
doc.core_properties.author=''
doc.core_properties.keywords='全唐诗;作者时期;LDA;人文学审读;B-C-D'
target=OUT/'QTS_BCD_Humanities_Review_Report.docx'
doc.save(target)

# 结构核验不能替代分页视觉核验；这里只检查完整同文与表格宽度。
from lxml import etree
with zipfile.ZipFile(target) as z:
    assert z.testzip() is None
    root=etree.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    actual=''.join(root.xpath('//w:body//w:t/text()',namespaces=ns))
    normal=lambda s:re.sub(r'\s','',s)
    assert normal(actual)==normal(''.join(expected)), 'Word正文未与Markdown逐块一致'
    for table in root.xpath('//w:tbl',namespaces=ns):
        widths=[int(x) for x in table.xpath('./w:tblGrid/w:gridCol/@w:w',namespaces=ns)]
        assert sum(widths)==9360
        assert table.xpath('./w:tblPr/w:tblW/@w:w',namespaces=ns)==['9360']
        for row in table.xpath('./w:tr',namespaces=ns):
            assert [int(x) for x in row.xpath('./w:tc/w:tcPr/w:tcW/@w:w',namespaces=ns)]==widths
    assert '\ufffd' not in actual
    result={'body_text_parity':True,'table_geometry_pass':True,'tables':len(doc.tables),'source_hyperlinks':len(root.xpath('//w:hyperlink',namespaces=ns)), 'numbered_lists':list_numbers, 'preset':'narrative_proposal', 'header_pattern':'editorial_cover', 'overrides':{'east_asia_body':'宋体','east_asia_headings':'等线','title_size':24,'table_font_size':9}, 'visual_qa':'UNAVAILABLE_LIBREOFFICE_SOFFICE_NOT_INSTALLED', 'render_attempt':'render_docx.py在启动soffice时返回WinError 2；未完成分页视觉核验。'}
    (HERE/'docx_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
