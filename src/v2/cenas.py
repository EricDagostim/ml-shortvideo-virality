"""Conta as cenas de um vídeo (cortes secos e fusões) e quantas delas são visualmente diferentes.

Método: amostra 4 quadros por segundo, compara o histograma HSV de cada quadro com o de meio segundo antes
(distância de Bhattacharyya) e marca um corte em cada pico acima do limiar, com no mínimo 1 s entre cortes.
Cenas "distintas": agrupa as cenas pelo histograma médio; uma cena só conta como nova se for diferente de todas
as anteriores.

Uso:
  calibrar <pasta videos/>                 compara com os planos anotados pela pipeline (plano_cenas.json)
  base <pasta v2> <pasta videos/>          baixa cada Short em baixa resolução, mede e apaga o vídeo
"""
import csv
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

FPS_AMOSTRA, JANELA, GAP_S = 4, 2, 1.0


def histogramas(caminho):
    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    passo = max(1, round(fps / FPS_AMOSTRA))
    hs, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % passo == 0:
            ok, q = cap.retrieve()
            if not ok:
                break
            hsv = cv2.cvtColor(cv2.resize(q, (90, 160)), cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256])
            hs.append(cv2.normalize(h, h).flatten())
        i += 1
    cap.release()
    return np.array(hs, dtype=np.float32), i / fps


def medir(caminho, limiar=0.25, limiar_distinta=0.45):
    hs, dur = histogramas(caminho)
    if len(hs) < 4:
        return None
    d = np.array([0.0] * JANELA + [cv2.compareHist(hs[k], hs[k - JANELA], cv2.HISTCMP_BHATTACHARYYA)
                                   for k in range(JANELA, len(hs))])
    cortes, ultimo = [], -1e9
    for k in range(1, len(d) - 1):
        if d[k] >= limiar and d[k] >= d[k - 1] and d[k] >= d[k + 1] and (k - ultimo) / FPS_AMOSTRA >= GAP_S:
            cortes.append(k)
            ultimo = k
    limites = [0] + cortes + [len(hs)]
    medias = [hs[a:b].mean(axis=0) for a, b in zip(limites[:-1], limites[1:]) if b > a]
    distintas = []
    for m in medias:
        if all(cv2.compareHist(m, x, cv2.HISTCMP_BHATTACHARYYA) > limiar_distinta for x in distintas):
            distintas.append(m)
    n = len(medias)
    return {"cen_cenas": n, "cen_cenas_min": n / dur * 60, "cen_plano_medio_s": dur / n,
            "cen_distintas": len(distintas), "cen_mudanca_media": float(d[JANELA:].mean()), "duracao_medida": dur}


def calibrar(videos):
    casos = []
    for plano in glob.glob(str(videos / "estudio-shorts" / "producoes" / "*" / "cenas" / "plano_cenas.json")):
        prod = Path(plano).parent.parent
        mp4 = prod / "render" / "final.mp4"
        if mp4.exists():
            casos.append((prod.name, len(json.load(open(plano))["planos"]), mp4))
    print(f"{len(casos)} produções com planos anotados")
    hist = {nome: histogramas(mp4) for nome, _, mp4 in casos}
    for limiar in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]:
        erros, linhas = [], []
        for nome, real, mp4 in casos:
            hs, dur = hist[nome]
            d = [0.0] * JANELA + [cv2.compareHist(hs[k], hs[k - JANELA], cv2.HISTCMP_BHATTACHARYYA) for k in range(JANELA, len(hs))]
            cortes, ultimo = 0, -1e9
            for k in range(1, len(d) - 1):
                if d[k] >= limiar and d[k] >= d[k - 1] and d[k] >= d[k + 1] and (k - ultimo) / FPS_AMOSTRA >= GAP_S:
                    cortes += 1
                    ultimo = k
            erros.append(cortes + 1 - real)
            linhas.append((nome, real, cortes + 1))
        reais = [r for _, r, _ in linhas]
        prev = [p for _, _, p in linhas]
        corr = np.corrcoef(reais, prev)[0, 1] if np.std(prev) > 0 else float("nan")
        print(f"limiar {limiar}: erro médio {np.mean(erros):+.1f}, erro absoluto {np.mean(np.abs(erros)):.1f}, correlação {corr:.2f}")
        if limiar == 0.35:
            for l in linhas:
                print("   ", l)


def base(S, videos):
    ids = [l.strip().rsplit("/", 1)[-1] for l in open(S / "ids_legendas.txt") if l.strip()]
    reg = json.load(open(videos / "estudio-shorts" / "analise" / "dados" / "videos.json"))["videos"]
    saida = S / "cenas.csv"
    feitos = {r["id"] for r in csv.DictReader(open(saida))} if saida.exists() else set()
    tmp = S / "vtmp"
    tmp.mkdir(exist_ok=True)
    campos = ["id", "cen_cenas", "cen_cenas_min", "cen_plano_medio_s", "cen_distintas", "cen_mudanca_media", "duracao_medida"]

    def um(vid):
        if vid in feitos:
            return None
        local = None
        if vid in reg and reg[vid].get("entrega"):
            achados = glob.glob(str(videos / reg[vid]["entrega"]) + "/*_1080x1920.mp4")
            local = achados[0] if achados else None
        if not local:
            subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "bv*[height<=640]/wv*/w", "-o", str(tmp / f"{vid}.%(ext)s"),
                            f"https://www.youtube.com/shorts/{vid}"], capture_output=True, timeout=180)
            achados = glob.glob(str(tmp / f"{vid}.*"))
            local = achados[0] if achados else None
        if not local:
            return {"id": vid}
        try:
            r = medir(local) or {}
        finally:
            if str(tmp) in local:
                os.remove(local)
        return {"id": vid, **r}

    novo = not saida.exists()
    with open(saida, "a", newline="") as f, ThreadPoolExecutor(3) as ex:
        w = csv.DictWriter(f, fieldnames=campos)
        if novo:
            w.writeheader()
        for r in ex.map(um, ids):
            if r:
                w.writerow({k: r.get(k) for k in campos})
                f.flush()
    print("cenas: fim")


if __name__ == "__main__":
    if sys.argv[1] == "calibrar":
        calibrar(Path(sys.argv[2]))
    else:
        base(Path(sys.argv[2]), Path(sys.argv[3]))
