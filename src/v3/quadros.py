"""Conteúdo ao longo do vídeo, sem baixar o vídeo: os 3 quadros que o YouTube gera a 25%, 50% e 75% da duração
(i.ytimg.com/vi/<id>/hq1.jpg, hq2.jpg, hq3.jpg — o mesmo CDN das capas, sem bloqueio de robô).

O quadro vem em 480x360 com o Short no meio e laterais desfocadas; recortamos só a faixa vertical do centro.
No vídeo final do Gabriel, antes de postar, pegamos os quadros nos mesmos pontos e passamos pelo mesmo recorte.

Features: vetores CLIP (média dos 3 quadros, em quadros_yt.npz), nota "animado", diversidade visual entre os quadros
(quanto o vídeo muda), brilho, saturação, colorido, bordas (detalhe e texto na tela, no topo e no meio) e quanto a capa
se parece com o conteúdo.

Uso: .venv/bin/python -m src.v3.quadros [--max 120000] [--paralelo 12]
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.v3.config import EMB, V3

SAIDA = V3 / "quadros_feats.jsonl"
ARQ_EMB = EMB / "quadros_yt.npz"


def recorte_central(im: Image.Image) -> Image.Image:
    """Faixa 9:16 do centro de um quadro 4:3 (o Short sem as laterais desfocadas)."""
    w, h = im.size
    if h > w:                       # já vertical (quadro tirado do vídeo local)
        return im.resize((round(360 * w / h), 360))
    cw = round(h * 9 / 16)
    x0 = (w - cw) // 2
    return im.crop((x0, 0, x0 + cw, h))


def quadros_do_video_local(video, duracao=None):
    """Quadros a 25/50/75% do MP4 local, já no formato dos hq1/2/3 (vertical 202x360, JPEG)."""
    import subprocess
    from src.v3.prever import duracao_video
    dur = duracao or duracao_video(video) or 60
    out = []
    for frac in (0.25, 0.5, 0.75):
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{dur * frac:.2f}", "-i", str(video), "-frames:v", "1", "-vf",
                            "scale=202:360", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "5", "-"], capture_output=True)
        if r.stdout:
            out.append(Image.open(BytesIO(r.stdout)).convert("RGB"))
    return out


def stats(ims, capa_vet=None, vets=None):
    f = {}
    br, sa, co, bo, bt, bm = [], [], [], [], [], []
    for im in ims:
        q = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(q, cv2.COLOR_BGR2HSV)
        g = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
        b, gg, r = [c.astype(np.float32) for c in cv2.split(q)]
        rg, yb = r - gg, 0.5 * (r + gg) - b
        e = cv2.Canny(g, 100, 200) > 0
        n = e.shape[0]
        br.append(hsv[..., 2].mean() / 255)
        sa.append(hsv[..., 1].mean() / 255)
        co.append(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))
        bo.append(e.mean())
        bt.append(e[: n // 3].mean())
        bm.append(e[n // 3: 2 * n // 3].mean())
    f.update({"qd_brilho": float(np.mean(br)), "qd_saturacao": float(np.mean(sa)), "qd_colorido": float(np.mean(co)),
              "qd_bordas": float(np.mean(bo)), "qd_bordas_topo": float(np.mean(bt)), "qd_bordas_meio": float(np.mean(bm)),
              "qd_brilho_var": float(np.std(br))})
    if vets is not None and len(vets) >= 2:
        S = vets @ vets.T
        f["qd_diversidade"] = float(1 - S[np.triu_indices(len(vets), 1)].mean())
    if vets is not None and capa_vet is not None and not np.isnan(capa_vet).any():
        f["qd_capa_sim"] = float((vets @ capa_vet).mean())
    return f


def _baixar(vid):
    ims = []
    for k in (1, 2, 3):
        try:
            d = urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/hq{k}.jpg", timeout=20).read()
            ims.append(recorte_central(Image.open(BytesIO(d)).convert("RGB")))
        except Exception:  # noqa: BLE001
            pass
    return vid, ims


def coletar(maximo, paralelo):
    from src.v3 import embeddings as emb
    b = pd.read_parquet(V3 / "base_v3.parquet", columns=["id", "canal_id", "publicado", "no_treino"])
    b = b[b["no_treino"] & (b["publicado"] >= pd.Timestamp("2022-01-01", tz="UTC"))].sort_values("publicado", ascending=False)
    b = b.assign(_r=b.groupby("canal_id").cumcount()).sort_values(["_r", "canal_id"])
    feitos = {json.loads(l)["id"] for l in open(SAIDA)} if SAIDA.exists() else set()
    ids = [i for i in b["id"] if i not in feitos][:maximo]
    capas, tem_c, _ = emb.carregar("capas", ids)
    pos = {i: k for k, i in enumerate(ids)}
    print(f"{len(ids)} Shorts ({len(feitos)} já feitos)", flush=True)
    z = dict(np.load(ARQ_EMB, allow_pickle=True)) if ARQ_EMB.exists() else {"ids": np.array([], object), "media": np.zeros((0, 512), np.float32)}
    ids_e, med_e = list(z["ids"]), [z["media"]]
    t0 = time.time()
    with open(SAIDA, "a") as f, ThreadPoolExecutor(paralelo) as ex:
        for k in range(0, len(ids), 1024):
            lote = list(ex.map(_baixar, ids[k:k + 1024]))
            ims = [im for _, l in lote for im in l]
            V, a = emb.emb_imagens(ims) if ims else (np.zeros((0, 512)), np.zeros(0))
            j = 0
            for vid, l in lote:
                n = len(l)
                if n == 0:
                    f.write(json.dumps({"id": vid, "erro": "sem_quadros"}) + "\n")
                    continue
                vets = V[j:j + n]
                cv = capas[pos[vid]] if tem_c[pos[vid]] else None
                linha = {"id": vid, "qd_n": n, "qd_animado": float(a[j:j + n].mean()), **stats(l, cv, vets)}
                f.write(json.dumps(linha) + "\n")
                ids_e.append(vid)
                m = vets.mean(0)
                med_e.append((m / np.linalg.norm(m))[None])
                j += n
            f.flush()
            tmp = ARQ_EMB.with_name("quadros_yt.tmp.npz")      # grava e troca de uma vez: quem lê nunca pega pela metade
            np.savez(tmp, ids=np.array(ids_e, dtype=object), media=np.concatenate(med_e).astype(np.float32))
            tmp.replace(ARQ_EMB)
            med_e = [np.concatenate(med_e)]
            print(f"{k + len(lote)}/{len(ids)} {(k + len(lote)) / (time.time() - t0):.1f}/s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=120000)
    ap.add_argument("--paralelo", type=int, default=12)
    a = ap.parse_args()
    coletar(a.max, a.paralelo)
