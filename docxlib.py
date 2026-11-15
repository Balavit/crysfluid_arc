"""Minimal LNCS (Springer splnproc2311) document builder: uses the template's own styles; strips macros."""
import re, os, shutil, zipfile, struct
from xml.sax.saxutils import escape

TPL = '/tmp/tpl'
TEXTW = 6918   # dxa
EMU_W = 4392930  # 12.2 cm, matches the template figure width

def _r(text, b=False, i=False, sup=False, sub=False, sz=None):
    pr = ''
    if b: pr += '<w:b/>'
    if i: pr += '<w:i/>'
    if sz: pr += f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>'
    if sup: pr += '<w:vertAlign w:val="superscript"/>'
    if sub: pr += '<w:vertAlign w:val="subscript"/>'
    return f'<w:r>{"<w:rPr>"+pr+"</w:rPr>" if pr else ""}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'

TOK = re.compile(r'(\*\*.+?\*\*|\*.+?\*|\^.+?\^|~.+?~)')
def runs(text, sz=None):
    out = []
    for part in TOK.split(text):
        if not part: continue
        if part.startswith('**'): out.append(_r(part[2:-2], b=True, sz=sz))
        elif part.startswith('*'): out.append(_r(part[1:-1], i=True, sz=sz))
        elif part.startswith('^'): out.append(_r(part[1:-1], sup=True, sz=sz))
        elif part.startswith('~'): out.append(_r(part[1:-1], sub=True, sz=sz))
        else: out.append(_r(part, sz=sz))
    return ''.join(out)

def para(text, style=None, ppr=''):
    st = f'<w:pStyle w:val="{style}"/>' if style else ''
    pp = f'<w:pPr>{st}{ppr}</w:pPr>' if (st or ppr) else ''
    return f'<w:p>{pp}{runs(text)}</w:p>'

def h1(t): return para(t, 'heading1')
def h2(t): return para(t, 'heading2')
def p1a(t): return para(t, 'p1a')          # first paragraph of a section: not indented
def p(t): return para(t)                   # normal body (indented)
def bullet(t): return para(t, 'bulletitem')
def numitem(t): return para(t, 'numitem')
def ref(t): return para(t, 'referenceitem')

def equation(tex, num):
    return (f'<w:p><w:pPr><w:pStyle w:val="equation"/></w:pPr><w:r><w:tab/></w:r>{runs(tex)}'
            f'<w:r><w:tab/><w:t>({num})</w:t></w:r></w:p>')

def table(caption, rows, widths, header=1, sz=16, align=None):
    """rows: list of list of str. widths: dxa per column (sum <= TEXTW)."""
    cap = f'<w:p><w:pPr><w:pStyle w:val="tablecaption"/><w:keepNext/></w:pPr>{runs(caption)}</w:p>'
    tot = sum(widths)
    x = [f'<w:tbl><w:tblPr><w:tblW w:w="{tot}" w:type="dxa"/><w:jc w:val="center"/><w:tblLayout w:type="fixed"/>'
         '<w:tblCellMar><w:left w:w="50" w:type="dxa"/><w:right w:w="50" w:type="dxa"/></w:tblCellMar>'
         '<w:tblLook w:val="0000"/></w:tblPr><w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths) + '</w:tblGrid>']
    n = len(rows)
    for ri, row in enumerate(rows):
        x.append('<w:tr><w:trPr><w:cantSplit/><w:jc w:val="center"/></w:trPr>')
        for ci, cell in enumerate(row):
            bd = ''
            if ri == 0: bd += '<w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            if ri == header - 1: bd += '<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
            if ri == n - 1: bd += '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
            jc = 'left' if (ci == 0 or (align and align[ci] == 'l')) else 'center'
            txt = cell if ri >= header else (cell if cell.startswith('**') else f'**{cell}**') if cell else cell
            x.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/><w:tcBorders>{bd}</w:tcBorders></w:tcPr>'
                     f'<w:p><w:pPr>{"" if (ri == n - 1 or n > 3) else "<w:keepNext/>"}<w:spacing w:before="20" w:after="20"/><w:ind w:firstLine="0"/><w:jc w:val="{jc}"/></w:pPr>{runs(txt, sz=sz)}</w:p></w:tc>')
        x.append('</w:tr>')
    x.append('</w:tbl>')
    after = '<w:p><w:pPr><w:spacing w:before="0" w:after="60"/><w:ind w:firstLine="0"/></w:pPr></w:p>'
    return cap + ''.join(x) + after

_figs = []   # (rid, filename, path)
def figure(png, caption, width_frac=1.0):
    if not os.path.exists(png): return para('[figure pending: ' + os.path.basename(png) + ']')
    from PIL import Image
    im = Image.open(png); w, h = im.size
    cx = int(EMU_W * width_frac); cy = int(cx * h / w)
    k = len(_figs) + 1; rid = f'rIdFig{k}'; fn = f'fig{k}.png'; _figs.append((rid, fn, png))
    d = (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="200" w:after="40"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
         f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{100+k}" name="Figure {k}"/>'
         '<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
         '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
         '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:nvPicPr><pic:cNvPr id="0" name="' + fn + '"/><pic:cNvPicPr/></pic:nvPicPr>'
         f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
         f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
         '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')
    cp = f'<w:p><w:pPr><w:pStyle w:val="figurecaption"/><w:jc w:val="both"/></w:pPr>{runs(caption)}</w:p>'
    return d + cp

def front(title, authors, affil, abstract, keywords):
    x = [para(title, 'papertitle'), para(authors, 'author'), para(affil, 'address')]
    x.append(f'<w:p><w:pPr><w:pStyle w:val="abstract"/></w:pPr>{_r("Abstract. ", b=True)}{runs(abstract)}</w:p>')
    x.append(f'<w:p><w:pPr><w:pStyle w:val="keywords"/></w:pPr>{_r("Keywords: ", b=True)}{runs(keywords)}</w:p>')
    return ''.join(x)

def build(body_xml, out_docx, header_left, header_right, workdir='/tmp/build_docx'):
    if os.path.exists(workdir): shutil.rmtree(workdir)
    shutil.copytree(TPL, workdir, ignore=shutil.ignore_patterns('t.zip'))
    W = workdir
    d = open(f'{W}/word/document.xml', encoding='utf-8-sig').read()
    prefix = d[:d.index('<w:body>') + len('<w:body>')]
    sect = re.findall(r'<w:sectPr.*?</w:sectPr>', d, flags=re.S)[-1]
    open(f'{W}/word/document.xml', 'w', encoding='utf-8').write(prefix + body_xml + sect + '</w:body></w:document>')
    # strip macros / customUI / sample chart
    for pth in ['word/vbaProject.bin', 'word/vbaData.xml', 'word/_rels/vbaProject.bin.rels']:
        if os.path.exists(f'{W}/{pth}'): os.remove(f'{W}/{pth}')
    shutil.rmtree(f'{W}/customUI', ignore_errors=True); shutil.rmtree(f'{W}/word/charts', ignore_errors=True)
    ct = open(f'{W}/[Content_Types].xml', encoding='utf-8-sig').read()
    ct = ct.replace('application/vnd.ms-word.document.macroEnabled.main+xml', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml')
    ct = re.sub(r'\s*<Default Extension="bin"[^>]*/>', '', ct)
    ct = re.sub(r'\s*<Override PartName="/word/vbaData.xml"[^>]*/>', '', ct)
    ct = re.sub(r'\s*<Override PartName="/word/charts/chart1.xml"[^>]*/>', '', ct)
    ct = ct.replace('ContentType="image/.png"', 'ContentType="image/png"')
    ct = re.sub(r'\s*<Default Extension="ICO"[^>]*/>', '', ct)
    open(f'{W}/[Content_Types].xml', 'w', encoding='utf-8').write(ct)
    rl = open(f'{W}/_rels/.rels', encoding='utf-8-sig').read()
    rl = re.sub(r'<Relationship Id="R2f9a12ed56c04698"[^>]*/>', '', rl); open(f'{W}/_rels/.rels', 'w', encoding='utf-8').write(rl)
    dr = open(f'{W}/word/_rels/document.xml.rels', encoding='utf-8-sig').read()
    dr = re.sub(r'<Relationship Id="rId1"[^>]*vbaProject[^>]*/>', '', dr)
    dr = re.sub(r'<Relationship Id="rId9"[^>]*chart[^>]*/>', '', dr)
    os.makedirs(f'{W}/word/media', exist_ok=True)
    add = ''
    for rid, fn, path in _figs:
        shutil.copy(path, f'{W}/word/media/{fn}')
        add += f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{fn}"/>'
    dr = dr.replace('</Relationships>', add + '</Relationships>'); open(f'{W}/word/_rels/document.xml.rels', 'w', encoding='utf-8').write(dr)
    # running heads
    for f, old, new in [('header1.xml', 'F. Author and S. Author', header_left), ('header2.xml', 'Contribution Title (shortened if too long) ', header_right)]:
        s = open(f'{W}/word/{f}', encoding='utf-8-sig').read(); s = s.replace(old, escape(new)); open(f'{W}/word/{f}', 'w', encoding='utf-8').write(s)
    # tighter vertical spacing to avoid large gaps around figures/headings
    st = open(f'{W}/word/styles.xml', encoding='utf-8-sig').read()
    def _sp(sid, old, new):
        nonlocal st
        m = re.search(r'<w:style [^>]*w:styleId="%s".*?</w:style>' % sid, st, re.S)
        if m: st = st.replace(m.group(0), m.group(0).replace(old, new, 1))
    _sp('heading1', 'w:before="360"', 'w:before="240"'); _sp('heading2', 'w:before="360"', 'w:before="200"')
    _sp('figurecaption', 'w:after="240"', 'w:after="100"')
    _sp('referenceitem', 'w:line="220"', 'w:line="200"'); _sp('referenceitem', 'w:val="18"', 'w:val="16"')
    open(f'{W}/word/styles.xml', 'w', encoding='utf-8').write(st)
    # docProps: neutral title
    if os.path.exists(out_docx): os.remove(out_docx)
    with zipfile.ZipFile(out_docx, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(f'{W}/[Content_Types].xml', '[Content_Types].xml')
        for root, _, files in os.walk(W):
            for fn in files:
                full = os.path.join(root, fn); arc = os.path.relpath(full, W)
                if arc == '[Content_Types].xml': continue
                z.write(full, arc)
    return out_docx
