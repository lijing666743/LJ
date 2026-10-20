"""Build and check the user-requested proposal presentation revision artifacts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import posixpath
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as ET

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'docs/presentations/proposal_revision_20261015'
MATERIAL = ROOT / 'sections/defense_20261015'
AUDIT = ROOT / 'metadata/audits/proposal_revision_20261015'
NS = {'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
FINAL = PROJECT / 'exports/李静_开题答辩_严谨修订可编辑版.pptx'

def read_content():
    return json.loads((MATERIAL / 'revision_content.json').read_text(encoding='utf-8'))

def prepare_plan():
    content = read_content()
    library = json.loads((PROJECT/'analysis/slide_library.json').read_text(encoding='utf-8'))
    slides=[]
    rationale=[]
    for item, source in zip(content['slides'],library['slides']):
        available={s['slot_id'] for s in source['slots']}
        replacements=[]
        for key,text in item['replacements'].items():
            slot=f"s{item['n']:02d}_sh{key}"
            assert slot in available,slot
            replacements.append({'slot_id':slot,'text':text})
        slides.append({'source_slide':item['n'],'purpose':item['title'],
                       'replacements':replacements})
        rationale.append({'target_slide':item['n'],'source_slide':item['n'],
            'layout_reason':'保留用户原顺序和底板，在现有文字、公式、框图和图片区域内修订',
            'semantic_purpose':item['title']})
    plan={'schema':'template_fill_pptx_plan.v1','source_pptx':str(PROJECT/'sources/original.pptx'),'slides':slides}
    (PROJECT/'analysis/fill_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    (PROJECT/'analysis/page_layout_rationale.json').write_text(json.dumps(rationale,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'slides':len(slides),'spoken_characters':sum(len(x['notes']) for x in content['slides']),
                      'planned_seconds':sum(x['seconds'] for x in content['slides'])}))

def shape_by_id(root,shape_id):
    for shape in root.findall('.//p:sp',NS):
        nv=shape.find('p:nvSpPr/p:cNvPr',NS)
        if nv is not None and nv.get('id')==str(shape_id):return shape
    raise KeyError(shape_id)

def apply_rich_runs(shape, parts, size_pt=21):
    """Write typography-correct, editable equation runs without rasterization."""
    tx=shape.find('p:txBody',NS)
    old=tx.find('a:p',NS)
    ppr=deepcopy(old.find('a:pPr',NS)) if old is not None and old.find('a:pPr',NS) is not None else ET.Element('{%s}pPr'%NS['a'])
    ppr.set('algn','ctr')
    for p in list(tx.findall('a:p',NS)):tx.remove(p)
    p=ET.SubElement(tx,'{%s}p'%NS['a']);p.append(ppr)
    for text,offset in parts:
        r=ET.SubElement(p,'{%s}r'%NS['a'])
        rpr=ET.SubElement(r,'{%s}rPr'%NS['a'],sz=str(round(size_pt*(.72 if offset else 1)*100)),lang='en-US')
        if offset:rpr.set('baseline',str(offset))
        solid=ET.SubElement(rpr,'{%s}solidFill'%NS['a']);ET.SubElement(solid,'{%s}srgbClr'%NS['a'],val='202020')
        ET.SubElement(rpr,'{%s}latin'%NS['a'],typeface='Arial')
        t=ET.SubElement(r,'{%s}t'%NS['a']);t.text=text

def set_font_size(shape, size_pt):
    for rpr in shape.findall('.//a:rPr',NS)+shape.findall('.//a:defRPr',NS)+shape.findall('.//a:endParaRPr',NS):
        rpr.set('sz',str(round(size_pt*100)))

def ordered_slide_paths(parts):
    rels=ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    targets={x.get('Id'):x.get('Target') for x in rels}
    root=ET.fromstring(parts['ppt/presentation.xml'])
    return [posixpath.normpath(posixpath.join('ppt',targets[x.get('{%s}id'%NS['r'])]))
            for x in root.findall('p:sldIdLst/p:sldId',NS)]

def finish_deck(draft):
    with ZipFile(draft) as z:parts={name:z.read(name) for name in z.namelist()}
    with ZipFile(PROJECT/'analysis/diagrams.pptx') as z:diagrams=[ET.fromstring(z.read(f'ppt/slides/slide{i}.xml')) for i in [1,2,3,4]]
    # Keep the source branding and layout. Remove only visible editing residue.
    removed={4:[2],5:[2],7:[3]}
    for i,name in enumerate(ordered_slide_paths(parts),1):
        root=ET.fromstring(parts[name])
        for sid in removed.get(i,[]):
            sp=shape_by_id(root,sid);sp.getparent().remove(sp)
        # Clear template title/body inline color fragments on changed text while
        # retaining the first source paragraph properties and font family.
        if i==10:
            apply_rich_runs(shape_by_id(root,53),[('T',0),('n',-25000),(' = (t',0),('n',-25000),('cmp',30000),(' − t',0),('n',-25000),('arr',30000),(') Δt',0)],21)
            set_font_size(shape_by_id(root,54),15.5)
            set_font_size(shape_by_id(root,55),11.5)
        if i==11:
            apply_rich_runs(shape_by_id(root,52),[('max   Λ',0),('succ',-25000),('(π) = E',0),('π',-25000),('[Σ',0),('t',-25000),(' N',0),('on',-25000),('(t)] / T',0)],22)
            set_font_size(shape_by_id(root,53),13.5)
        if i in [15,16]:
            # The existing short header frame is widened into unused header
            # space; position, typeface, size, and logo remain unchanged.
            shape_by_id(root,47).find('p:spPr/a:xfrm/a:ext',NS).set('cx',str(500*9525))
        if i==15:
            for sid in [52,56,60,64]:set_font_size(shape_by_id(root,sid),16)
            # Use the body font's existing dark blue for the concise sublines.
            for sid in [52,56,60,64]:
                for color in shape_by_id(root,sid).findall('.//a:srgbClr',NS):color.set('val','104495')
        if i==19:
            # The original exclamation mark wrapped into the subtitle area.
            set_font_size(shape_by_id(root,27),60)
        if i in [4,5,8,10]:
            tree=root.find('p:cSld/p:spTree',NS)
            picture=tree.find('p:pic',NS)
            tree.remove(picture)
            incoming=diagrams[{4:0,5:1,10:2,8:3}[i]].find('p:cSld/p:spTree',NS)
            shapes=[deepcopy(x) for x in incoming if ET.QName(x).localname in ['sp','cxnSp','grpSp']]
            # The exporter places connectors first. Put the opaque background
            # behind them so link arrows stay visible in native PowerPoint.
            def stacking_order(obj):
                nv=obj.find('.//p:cNvPr',NS)
                if nv is not None and (nv.get('name','').endswith('_frame') or 'scene8_layer' in nv.get('name','')):return 0
                return 1 if ET.QName(obj).localname=='cxnSp' else 2
            shapes.sort(key=stacking_order)
            maxid=max(int(x.get('id')) for x in root.findall('.//p:cNvPr',NS))
            remap={}
            for obj in shapes:
                for nv in obj.findall('.//p:cNvPr',NS):
                    maxid+=1;remap[nv.get('id')]=str(maxid);nv.set('id',str(maxid))
            for obj in shapes:
                for con in obj.findall('.//a:stCxn',NS)+obj.findall('.//a:endCxn',NS):
                    con.set('id',remap[con.get('id')])
                tree.append(obj)
        parts[name]=ET.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    # The user requested a separate document, with no narration in PPT notes.
    for name in parts:
        if re.fullmatch(r'ppt/notesSlides/notesSlide\d+.xml',name):
            root=ET.fromstring(parts[name])
            for sp in root.findall('.//p:sp',NS):
                ph=sp.find('p:nvSpPr/p:nvPr/p:ph',NS)
                if ph is not None and ph.get('type')=='body':
                    for text_node in sp.findall('.//a:t',NS):text_node.text=''
            parts[name]=ET.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    FINAL.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(FINAL,'w',ZIP_DEFLATED) as z:
        for name,data in parts.items():z.writestr(name,data)
    print(FINAL)

def make_documents():
    from docx import Document
    from docx.shared import Cm,Pt,RGBColor
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    data=read_content()
    doc=Document();sec=doc.sections[0]
    sec.page_width=Cm(21);sec.page_height=Cm(29.7)
    sec.top_margin=Cm(1.9);sec.bottom_margin=Cm(1.8);sec.left_margin=Cm(2.2);sec.right_margin=Cm(2.2)
    for style_name,size in [('Normal',11),('Title',19),('Subtitle',10),('Heading 1',14),('Heading 2',11.5)]:
        st=doc.styles[style_name];st.font.name='微软雅黑';st.font.size=Pt(size);st.font.color.rgb=RGBColor(0,0,0)
        st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'微软雅黑')
        st.paragraph_format.space_after=Pt(5)
        for border in st.element.findall('.//'+qn('w:pBdr')):border.getparent().remove(border)
    doc.styles['Normal'].paragraph_format.line_spacing=1.25
    doc.styles['Heading 1'].paragraph_format.space_before=Pt(9)
    doc.styles['Heading 2'].paragraph_format.space_before=Pt(6)
    title=doc.add_paragraph('开题答辩逐页讲稿','Title')
    doc.add_paragraph(data['title'],'Subtitle')
    doc.add_paragraph('答辩人 李静    导师 杨恒    对应修订版 19 页 PPT','Subtitle')
    doc.add_paragraph('正文按约 9 分钟编排。章节过渡页一句带过，重点讲第 10—12 页。建议按每页的意思自然讲述，无需逐字读公式。后附问题回答供答辩准备，不计入汇报时间。')
    for item in data['slides']:
        doc.add_paragraph(f"第 {item['n']} 页  {item['title']}",'Heading 2')
        doc.add_paragraph(item['notes'])
    doc.add_paragraph('答辩问题准备','Heading 1')
    doc.add_paragraph('回答以现有模型和证据为准。涉及结果时，应明确开发评价、预算和训练种子边界。')
    for index,qa in enumerate(data['qa'],1):
        doc.add_paragraph(f"{index}  {qa['q']}",'Heading 2');doc.add_paragraph(qa['a'])
    footer=sec.footer.paragraphs[0];footer.alignment=2
    footer.add_run('李静 开题答辩讲稿  ')
    f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
    doc.core_properties.title='李静开题答辩逐页讲稿'
    doc.core_properties.subject=data['title'];doc.core_properties.author='李静'
    output=MATERIAL/'李静_开题答辩逐页讲稿_8至10分钟.docx';doc.save(output)
    md=['# 开题答辩逐页讲稿','',data['title'],'','正文按约9分钟编排，与修订版19页PPT逐页对应。问题准备不计入汇报时长。','']
    for item in data['slides']:md += [f"## 第{item['n']}页 {item['title']}",'',item['notes'],'']
    md+=['## 答辩问题准备','']
    for qa in data['qa']:md += [f"### {qa['q']}",'',qa['a'],'']
    (MATERIAL/'逐页答辩稿.md').write_text('\n'.join(md),encoding='utf-8')
    print(output)

def validate():
    data=read_content();report={'checks':{},'artifacts':{}}
    with ZipFile(FINAL) as z:
        parts={name:z.read(name) for name in z.namelist()}
        slides=ordered_slide_paths(parts)
        assert len(slides)==19
        report['checks']['slide_count_19']=True
        slide_text={}
        for i,slide_path in enumerate(slides,1):
            root=ET.fromstring(z.read(slide_path))
            ids=[x.get('id') for x in root.findall('.//p:cNvPr',NS)];assert len(ids)==len(set(ids))
            for x in root.findall('.//a:stCxn',NS)+root.findall('.//a:endCxn',NS):assert x.get('id') in ids
            slide_text[i]=''.join(root.xpath('//a:t/text()',namespaces=NS))
            for obj in root.findall('.//p:sp',NS):
                name=obj.find('p:nvSpPr/p:cNvPr',NS).get('name','')
                if name.startswith('scene'):
                    assert obj.find('p:spPr',NS) is not None
            rel_path=posixpath.join(posixpath.dirname(slide_path),'_rels',posixpath.basename(slide_path)+'.rels')
            rels=ET.fromstring(z.read(rel_path))
            target=next((x.get('Target') for x in rels if x.get('Type').endswith('/notesSlide')),None)
            if target:
                notes_path=posixpath.normpath(posixpath.join(posixpath.dirname(slide_path),target))
                notes=ET.fromstring(z.read(notes_path))
                note=''.join(notes.xpath('//a:t/text()',namespaces=NS))
                assert data['slides'][i-1]['notes'] not in note,(i,note)
                for sp in notes.findall('.//p:sp',NS):
                    ph=sp.find('p:nvSpPr/p:nvPr/p:ph',NS)
                    if ph is not None and ph.get('type')=='body':assert not ''.join(sp.xpath('.//a:t/text()',namespaces=NS))
        alltext='\n'.join(slide_text.values())
        for residue in ['标红','一区不用','研究法方法','最优策略','CPU 利用率与计算份额','地面/机载任务','手机','小车']:
            assert residue not in alltext,residue
        for i in [4,5,10]:
            assert 'UAV 4' in slide_text[i]
            root=ET.fromstring(z.read(slides[i-1]))
            assert not root.findall('.//p:pic',NS),'unreplaced picture on scenario slide'
        assert not ET.fromstring(z.read(slides[7])).findall('.//p:pic',NS)
        report['checks'].update(native_diagrams=True,speaker_notes_empty=True,editing_residue_removed=True)
        (PROJECT/'validation/final_readback.json').write_text(json.dumps(slide_text,ensure_ascii=False,indent=2),encoding='utf-8')
        with ZipFile(PROJECT/'sources/original.pptx') as original:
            assert z.read('ppt/presentation.xml') != b''
            original_size=ET.fromstring(original.read('ppt/presentation.xml')).find('p:sldSz',NS)
            final_size=ET.fromstring(z.read('ppt/presentation.xml')).find('p:sldSz',NS)
            assert dict(original_size.attrib)==dict(final_size.attrib)
            for name in original.namelist():
                if name.startswith(('ppt/slideMasters/','ppt/slideLayouts/','ppt/theme/')) and name in z.namelist():
                    # Template-fill rewrites relationship targets; the visual
                    # theme/master/layout XML itself must remain byte-identical.
                    if '/_rels/' not in name:assert original.read(name)==z.read(name),name
            report['checks']['original_master_layout_theme_preserved']=True
    for path in [FINAL,MATERIAL/'李静_开题答辩逐页讲稿_8至10分钟.docx']:
        report['artifacts'][str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (AUDIT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','finish','documents','validate']);parser.add_argument('--draft')
    args=parser.parse_args()
    if args.action=='prepare':prepare_plan()
    elif args.action=='finish':finish_deck(Path(args.draft))
    elif args.action=='documents':make_documents()
    else:validate()
