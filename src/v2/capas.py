"""Baixa a capa vertical (oar2.jpg) de cada Short e extrai features clássicas da imagem.

Uso: .venv/bin/python -I capas.py <pasta v2>   (lê ids_legendas.txt, grava capas/<id>.jpg e capas.csv)
"""
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

S = Path(sys.argv[1])
ids = [l.strip().rsplit("/", 1)[-1] for l in open(S / "ids_legendas.txt") if l.strip()]
PASTA = S / "capas"
PASTA.mkdir(exist_ok=True)
ROSTO = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def baixar(vid):
    destino = PASTA / f"{vid}.jpg"
    if destino.exists():
        return
    try:
        destino.write_bytes(urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/oar2.jpg", timeout=20).read())
    except Exception:
        pass


def features(vid):
    img = cv2.imread(str(PASTA / f"{vid}.jpg"))
    if img is None:
        return {"id": vid}
    img = cv2.resize(img, (360, 640))
    cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b, g, r = [c.astype(float) for c in cv2.split(img)]
    rg, yb = r - g, 0.5 * (r + g) - b                      # colorido de Hasler e Süsstrunk (2003)
    colorido = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())
    matiz, sat, val = hsv[..., 0].astype(float), hsv[..., 1] / 255, hsv[..., 2] / 255
    quente = ((matiz < 25) | (matiz > 160)) & (sat > 0.35) & (val > 0.3)   # vermelho, laranja, amarelo forte
    bordas = cv2.Canny(cinza, 100, 200) > 0
    rostos = ROSTO.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
    maior = max((w * h for (_, _, w, h) in rostos), default=0) / (360 * 640)
    # texto na capa (aproximação): densidade de bordas na faixa de cima e na do meio, onde vão títulos e legendas
    faixa_topo, faixa_meio = bordas[:213].mean(), bordas[213:427].mean()
    return {
        "id": vid,
        "capa_brilho": val.mean(),
        "capa_contraste": cinza.std() / 255,
        "capa_saturacao": sat.mean(),
        "capa_colorido": colorido,
        "capa_quente": quente.mean(),
        "capa_escuro": (val < 0.2).mean(),
        "capa_nitidez": np.log1p(cv2.Laplacian(cinza, cv2.CV_64F).var()),
        "capa_bordas": bordas.mean(),
        "capa_bordas_topo": faixa_topo,
        "capa_bordas_meio": faixa_meio,
        "capa_rostos": len(rostos),
        "capa_rosto_area": maior,
    }


with ThreadPoolExecutor(16) as ex:
    list(ex.map(baixar, ids))
tabela = pd.DataFrame([features(v) for v in ids])
tabela.to_csv(S / "capas.csv", index=False)
print(f"{tabela['capa_brilho'].notna().sum()} de {len(ids)} capas com features")
print(tabela.describe().T[["mean", "50%", "max"]].round(3))
