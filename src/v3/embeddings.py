"""Vetores de título (multilíngue) e de capa (CLIP), com cache em data/v3/embeddings.

Título: paraphrase-multilingual-mpnet-base-v2 — "E se o Sol sumisse?" e "What if the Sun disappeared?" ficam próximos,
        então um tema que estourou em inglês conta como evidência para o mesmo tema em português.
Capa:   CLIP ViT-B-32 (LAION-2B). Também dá a nota "animado" (zero-shot): animação/3D/desenho x filmagem real.

Uso: .venv/bin/python -m src.v3.embeddings   (calcula o que falta para a base em data/v3/base_v3.parquet)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.v3.config import CAPAS, EMB, V3

# Modelos já baixados: não consultar o Hugging Face a cada carga (se o site estiver lento, a carga trava)
if (Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2").exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

MODELO_TEXTO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODELO_CLIP = ("ViT-B-32", "laion2b_s34b_b79k")
ANIMADO = ["a 3D animation render", "a cartoon illustration", "computer generated 3D graphics", "a frame from an animated video",
           "a 2D animated drawing", "a digital illustration"]
REAL = ["a photo of a real person", "a real-life video frame filmed with a camera", "a photograph of a real place",
        "a selfie video", "a screenshot of a real video", "a person talking to the camera"]

_cache = {}


def _dispositivo():
    import torch
    # 1 thread de CPU no torch: com o LightGBM no mesmo processo, os dois OpenMP brigam e o torch.randn trava
    # (macOS, 13/11/2026). A conta pesada roda na GPU (MPS) de qualquer forma.
    torch.set_num_threads(1)
    return "mps" if torch.backends.mps.is_available() else "cpu"


def modelo_texto():
    if "texto" not in _cache:
        modelo_clip()   # cria o CLIP antes (ordem que funciona com o OpenMP; ver _dispositivo)
        from sentence_transformers import SentenceTransformer
        _cache["texto"] = SentenceTransformer(MODELO_TEXTO, device=_dispositivo())
    return _cache["texto"]


def modelo_clip():
    if "clip" not in _cache:
        _dispositivo()   # fixa 1 thread no torch ANTES de criar o modelo
        import open_clip
        import torch
        m, _, prep = open_clip.create_model_and_transforms(MODELO_CLIP[0], pretrained=MODELO_CLIP[1])
        m = m.to(_dispositivo()).eval()
        tok = open_clip.get_tokenizer(MODELO_CLIP[0])
        with torch.no_grad():
            tx = m.encode_text(tok(ANIMADO + REAL).to(_dispositivo())).float()
            tx = tx / tx.norm(dim=-1, keepdim=True)
        _cache["clip"] = (m, prep, tx.cpu().numpy())
    return _cache["clip"]


def emb_titulos(titulos: list[str]) -> np.ndarray:
    import re
    limpos = [re.sub(r"#\w+", "", t or "").strip() or (t or "") for t in titulos]
    return modelo_texto().encode(limpos, batch_size=128, normalize_embeddings=True, show_progress_bar=False,
                                 convert_to_numpy=True).astype(np.float32)


def emb_imagens(imagens) -> tuple[np.ndarray, np.ndarray]:
    """imagens: lista de PIL.Image. Devolve (vetores normalizados, nota animado 0–1)."""
    import torch
    from concurrent.futures import ThreadPoolExecutor
    m, prep, tx = modelo_clip()
    vets = []
    with ThreadPoolExecutor(8) as ex:   # decodificar/redimensionar em paralelo (o PIL solta o GIL)
        prontas = list(ex.map(lambda im: prep(im.convert("RGB")), imagens))
    for k in range(0, len(prontas), 128):
        lote = torch.stack(prontas[k:k + 128]).to(_dispositivo())
        with torch.no_grad():
            v = m.encode_image(lote).float()
        vets.append((v / v.norm(dim=-1, keepdim=True)).cpu().numpy())
    V = np.concatenate(vets) if vets else np.zeros((0, 512), np.float32)
    logits = 100 * V @ tx.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    animado = p[:, :len(ANIMADO)].sum(1)
    return V.astype(np.float32), animado.astype(np.float32)


def _salvo(nome):
    arq = EMB / f"{nome}.npz"
    if not arq.exists():
        return {}
    z = np.load(arq, allow_pickle=True)
    return {"ids": list(z["ids"]), **{k: z[k] for k in z.files if k != "ids"}}


def atualizar_titulos(df: pd.DataFrame):
    EMB.mkdir(parents=True, exist_ok=True)
    s = _salvo("titulos")
    feitos = set(s.get("ids", []))
    novos = df[~df["id"].isin(feitos)].drop_duplicates("id")
    if len(novos):
        V = emb_titulos(novos["titulo"].tolist())
        ids = s.get("ids", []) + novos["id"].tolist()
        vets = np.concatenate([s["vetores"], V]) if s else V
        np.savez(EMB / "titulos.npz", ids=np.array(ids, dtype=object), vetores=vets)
    print(f"títulos: {len(novos)} novos, {len(feitos) + len(novos)} no total")


def atualizar_capas(df: pd.DataFrame):
    from PIL import Image
    from src.v3.quadros import recorte_central
    EMB.mkdir(parents=True, exist_ok=True)
    s = _salvo("capas")
    feitos = set(s.get("ids", []))
    alvo = [i for i in df["id"].drop_duplicates() if i not in feitos and (CAPAS / f"{i}.jpg").exists()]
    ids, vets, anim = s.get("ids", []), [s["vetores"]] if s else [], [s["animado"]] if s else []
    for k in range(0, len(alvo), 2048):
        def abrir(i):
            try:
                return i, recorte_central(Image.open(CAPAS / f"{i}.jpg").convert("RGB"))
            except Exception:  # noqa: BLE001
                return i, None
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(8) as ex:
            abertas = [(i, im) for i, im in ex.map(abrir, alvo[k:k + 2048]) if im is not None]
        lote_ids, imgs = [i for i, _ in abertas], [im for _, im in abertas]
        V, a = emb_imagens(imgs)
        ids += lote_ids
        vets.append(V)
        anim.append(a)
        np.savez(EMB / "capas.npz", ids=np.array(ids, dtype=object), vetores=np.concatenate(vets), animado=np.concatenate(anim))
        print(f"capas: {k + len(lote_ids)}/{len(alvo)}", flush=True)
    print(f"capas: {len(alvo)} novas, {len(ids)} no total")


def carregar_quadros(ids) -> tuple[np.ndarray, np.ndarray]:
    """Vetor CLIP médio dos 3 quadros do vídeo (quadros_yt.npz, de quadros.py) alinhado com `ids`."""
    arq = EMB / "quadros_yt.npz"
    M = np.full((len(ids), 512), np.nan, np.float32)
    tem = np.zeros(len(ids), bool)
    if not arq.exists():
        return M, tem
    import time
    for tentativa in range(20):          # o coletor regrava o arquivo a cada lote: tenta de novo se pegar no meio
        try:
            with np.load(arq, allow_pickle=True) as z:
                zi, zm = z["ids"], z["media"]
            break
        except (EOFError, OSError, ValueError, KeyError):
            time.sleep(3)
    else:
        raise RuntimeError(f"não consegui ler {arq}")
    pos = {i: k for k, i in enumerate(zi)}
    for r, i in enumerate(ids):
        k = pos.get(i)
        if k is not None:
            M[r] = zm[k]
            tem[r] = True
    return M, tem


def carregar(nome: str, ids) -> tuple[np.ndarray, np.ndarray]:
    """Matriz alinhada com `ids` (linhas sem vetor ficam com NaN) e máscara de quem tem vetor."""
    s = _salvo(nome)
    pos = {i: k for k, i in enumerate(s.get("ids", []))}
    dim = s["vetores"].shape[1] if s else {"titulos": 768, "capas": 512}[nome]
    M = np.full((len(ids), dim), np.nan, np.float32)
    tem = np.zeros(len(ids), bool)
    for r, i in enumerate(ids):
        k = pos.get(i)
        if k is not None:
            M[r] = s["vetores"][k]
            tem[r] = True
    extra = None
    if s and "animado" in s:
        extra = np.array([s["animado"][pos[i]] if i in pos else np.nan for i in ids], np.float32)
    return M, tem, extra


if __name__ == "__main__":
    base = pd.read_parquet(V3 / "base_v3.parquet", columns=["id", "titulo"])
    atualizar_titulos(base)
    atualizar_capas(base)
