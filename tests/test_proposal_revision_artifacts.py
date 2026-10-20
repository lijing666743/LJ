"""Independent OOXML acceptance checks for the proposal presentation artifacts."""
from pathlib import Path
import json
import posixpath
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import unittest

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'docs/presentations/proposal_revision_20261015'
PPTX=PROJECT/'exports/李静_开题答辩_原风格修订版.pptx'
MATERIAL=ROOT/'sections/defense_20261015'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def slides(z):
    rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read('ppt/_rels/presentation.xml.rels'))}
    pres=ET.fromstring(z.read('ppt/presentation.xml'))
    return [posixpath.normpath(posixpath.join('ppt',rels[s.get('{%s}id'%NS['r'])]))
            for s in pres.findall('p:sldIdLst/p:sldId',NS)]

def test_same_19_slides_and_unchanged_branding():
    with ZipFile(PPTX) as final,ZipFile(PROJECT/'sources/original.pptx') as source:
        assert len(slides(final))==len(slides(source))==19
        for name in source.namelist():
            if name.startswith(('ppt/slideMasters/','ppt/slideLayouts/','ppt/theme/')) and '/_rels/' not in name:
                assert final.read(name)==source.read(name)

def test_no_speech_in_notes():
    with ZipFile(PPTX) as z:
        for name in z.namelist():
            if name.startswith('ppt/notesSlides/') and name.endswith('.xml'):
                rt=ET.fromstring(z.read(name))
                for sp in rt.findall('.//p:sp',NS):
                    ph=sp.find('p:nvSpPr/p:nvPr/p:ph',NS)
                    if ph is not None and ph.get('type')=='body':
                        assert not ''.join(t.text or '' for t in sp.findall('.//a:t',NS))

def test_scene_images_keep_original_position_and_size(page):
    with ZipFile(PPTX) as z,ZipFile(PROJECT/'sources/original.pptx') as source:
        rt=ET.fromstring(z.read(slides(z)[page-1]))
        old=ET.fromstring(source.read(slides(source)[page-1]))
        pic=rt.find('.//p:pic/p:spPr/a:xfrm',NS)
        original=old.find('.//p:pic/p:spPr/a:xfrm',NS)
        assert ET.tostring(pic)==ET.tostring(original)
        assert len(rt.findall('.//p:sp/p:txBody',NS))>=len(old.findall('.//p:sp/p:txBody',NS))-1

def test_offloading_arrows_are_single_direction_and_one_wireless_hop(page,expected):
    with ZipFile(PPTX) as z:rt=ET.fromstring(z.read(slides(z)[page-1]))
    orange=[]
    for edge in rt.findall('.//p:cxnSp',NS):
        name=edge.find('p:nvCxnSpPr/p:cNvPr',NS).get('name','')
        if not name.startswith('proposal_single_hop'):continue
        line=edge.find('p:spPr/a:ln',NS)
        assert line.find('a:solidFill/a:srgbClr',NS).get('val')=='F28C28'
        assert line.find('a:headEnd',NS).get('type')=='none'
        assert line.find('a:tailEnd',NS).get('type')=='triangle'
        orange.append(name)
    assert len(orange)==expected

def test_no_existing_experiment_in_proposal():
    with ZipFile(PPTX) as z:
        text=''.join(t.text or '' for p in slides(z) for t in ET.fromstring(z.read(p)).findall('.//a:t',NS))
    for forbidden in ['A/B/C','当前实验','已有开发','40k','seed42','412','391','开发试验']:
        assert forbidden not in text

def test_random_arrival_arrows_point_toward_uav():
    with ZipFile(PPTX) as z:rt=ET.fromstring(z.read(slides(z)[9]))
    arrivals=[e for e in rt.findall('.//p:cxnSp',NS) if e.find('p:nvCxnSpPr/p:cNvPr',NS).get('name','').startswith('proposal_arrival_')]
    assert len(arrivals)==5
    for edge in arrivals:
        assert edge.find('p:spPr/a:xfrm',NS).get('flipV')=='1'
        assert edge.find('p:spPr/a:ln/a:headEnd',NS).get('type')=='none'
        assert edge.find('p:spPr/a:ln/a:tailEnd',NS).get('type')=='triangle'

def test_separate_speech_document_covers_all_19_slides():
    with ZipFile(MATERIAL/'李静_逐页答辩稿_开题版_8至10分钟.docx') as z:
        root=ET.fromstring(z.read('word/document.xml'))
        text=''.join(t.text or '' for t in root.findall('.//w:t',NS))
        assert root.find('.//w:pBdr',NS) is None
    content=json.loads((MATERIAL/'proposal_v2_content.json').read_text(encoding='utf-8'))
    for item in content['slides']:
        assert f"第 {item['n']} 页" in text
        assert item['notes'] in text
    assert 8*60<=sum(s['seconds'] for s in content['slides'])<=10*60


class ProposalArtifactTests(unittest.TestCase):
    test_branding = staticmethod(test_same_19_slides_and_unchanged_branding)
    test_notes = staticmethod(test_no_speech_in_notes)
    test_proposal_only = staticmethod(test_no_existing_experiment_in_proposal)
    test_arrival_directions = staticmethod(test_random_arrival_arrows_point_toward_uav)
    test_speech_document = staticmethod(test_separate_speech_document_covers_all_19_slides)

    def test_picture_geometry(self):
        for page in [4, 5, 8, 10]:
            with self.subTest(page=page):
                test_scene_images_keep_original_position_and_size(page)

    def test_single_hop_paths(self):
        for page in [4, 5, 10]:
            with self.subTest(page=page):
                test_offloading_arrows_are_single_direction_and_one_wireless_hop(page, 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
