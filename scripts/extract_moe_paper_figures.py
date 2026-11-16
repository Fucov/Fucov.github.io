#!/usr/bin/env python3
"""Archive selected original figures from the authors' arXiv HTML/PDF sources.

NO drawings, generated charts, reconstructed data, or hotlinked images.
The downloaded images are the authors' original figure artwork.
This script only selects, optionally joins existing author-provided subpanels,
and converts to PNG. Attribution stays in the Markdown.
"""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
from urllib.parse import urljoin
import re, json, sys, requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps
import fitz

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/"public/research/moe-paper-figures"
OUT.mkdir(parents=True,exist_ok=True)
PAPERS={
    "tempo":("2608.13057v2",[1,2]),
    "relief":("2611.11402v1",[1,5]),
    "megaflux":("2610.00671v1",[2]),
    "overlap":("2611.11857v1",[2,3]),
}
sess=requests.Session()
sess.headers.update({"User-Agent":"Mozilla/5.0 (research-notes; author figure archiver)", "Accept":"text/html,image/*,application/pdf;q=0.9,*/*;q=0.8"})

def grab(url):
    r=sess.get(url,timeout=65)
    r.raise_for_status()
    return r

def html_figures(arxiv, indices):
    url="https://arxiv.org/html/"+arxiv
    r=grab(url)
    soup=BeautifulSoup(r.text,"html.parser")
    output={}
    for n in indices:
        figure=None
        for f in soup.select("figure, .ltx_figure"):
            fid=(f.get("id") or "").lower()
            if re.search(r"\\.f"+str(n)+r"$",fid):
                figure=f;break
        if figure is None:
            for f in soup.select("figure, .ltx_figure"):
                c=f.select_one("figcaption, .ltx_caption")
                if c and re.search(r"\\b(?:Figure|Fig\\.)\\s*"+str(n)+r"\\b",c.get_text(" ",strip=True),flags=re.I):
                    figure=f;break
        if figure is None: continue
        urls=[]
        for img in figure.select("img"):
            src=img.get("src") or img.get("data-src")
            if not src and img.get("srcset"): src=img["srcset"].split(",")[0].split()[0]
            if src:
                u=urljoin(r.url,src)
                if u not in urls:urls.append(u)
        panels=[]
        for u in urls:
            try:
                data=grab(u).content
                im=Image.open(BytesIO(data))
                im.load()
                panels.append(ImageOps.exif_transpose(im).convert("RGB"))
            except Exception as ex:
                print("[WARN] image download:",u,repr(ex),flush=True)
        if panels:
            if len(panels)==1: output[n]=panels[0]
            else:
                # Join actual author panels without modifying their data.
                gap=8
                width=sum(x.width for x in panels)+(len(panels)-1)*gap
                height=max(x.height for x in panels)
                im=Image.new("RGB",(width,height),"white")
                pos=0
                for p in panels:
                    im.paste(p,(pos,0)); pos+=p.width+gap
                output[n]=im
    return output

def pdf_figures(arxiv,indices):
    """Fallback: crop the *actual PDF rendering* immediately above the figure caption."""
    raw=grab("https://arxiv.org/pdf/"+arxiv.split("v")[0]).content
    doc=fitz.open(stream=raw,filetype="pdf")
    result={}
    for page in doc:
        blocks=sorted([b for b in page.get_text("blocks") if isinstance(b[4],str)],key=lambda b:(b[1],b[0]))
        for n in indices:
            if n in result: continue
            for b in blocks:
                t=" ".join(b[4].split())
                if not re.search(r"^(Figure|Fig\\.)\\s*"+str(n)+r"\\s*[.:]",t,re.I):continue
                # Remove caption; keep artwork above it, limited to column-width/page.
                caption_top=float(b[1])
                x0=max(0,float(b[0])-8); x1=min(page.rect.width,float(b[2])+8)
                if x1-x0<page.rect.width*.30:
                    x0=0;x1=page.rect.width
                y0=max(0,caption_top-page.rect.height*.46)
                clip=fitz.Rect(x0,y0,x1,max(y0+40,caption_top-3))
                pix=page.get_pixmap(matrix=fitz.Matrix(1.7,1.7),clip=clip,alpha=False)
                result[n]=Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
                break
        if len(result)==len(indices):break
    return result

manifest={}
missing=[]
for key,(arxiv,nums) in PAPERS.items():
    try: fig=html_figures(arxiv,nums)
    except Exception as ex:
        print("[WARN] arXiv HTML:",key,repr(ex),flush=True);fig={}
    absent=[n for n in nums if n not in fig]
    if absent:
        try:
            fallback=pdf_figures(arxiv,absent)
            fig.update(fallback)
        except Exception as ex:
            print("[WARN] PDF fallback:",key,repr(ex),flush=True)
    for n in nums:
        name=f"{key}_fig{n}.png"
        if n not in fig:
            print("[MISSING]",name,flush=True)
            missing.append(name)
            continue
        p=OUT/name
        fig[n].save(p,format="PNG",optimize=True)
        print("[OK]",p,p.stat().st_size,flush=True)
        manifest[name]={"source":"https://arxiv.org/abs/"+arxiv.split("v")[0],"figure":n,"note":"Unmodified author-provided figure or crop of original PDF figure"}
(OUT/"FIGURE_SOURCES.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
print(f"Downloaded {len(manifest)} / 7 official figures")
if missing: sys.exit("Missing original figures: "+", ".join(missing))
