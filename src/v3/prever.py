"""Veredito ANTES de postar: este Short vai estourar no canal (>= 5x a régua)?

Entrada: a pasta da entrega (post.json + vídeo + capa) ou título/vídeo/capa soltos, a hora planejada e o canal.
O histórico do canal vem da YouTube Data API (Shorts públicos, views atuais) e a "demanda" vem da base v3 (títulos
parecidos de outros canais, publicados há pelo menos 14 dias).

Saída: probabilidade calibrada de estourar, veredito (VIRAL / PROMISSOR / NEUTRO / NÃO VIRAL), o que puxou a nota para
cima e para baixo e os títulos parecidos de outros canais com o resultado que tiveram.

Uso:
  .venv/bin/python -m src.v3.prever --entrega ../videos/entregas/simulando/2026-11-08_sm12-sol-some
  .venv/bin/python -m src.v3.prever --canal simulando --titulo "E se a Lua caísse?" --video final.mp4
  .venv/bin/python -m src.v3.prever --canal simulando --titulo "A" --titulo "B"        (compara títulos, sem vídeo)
  .venv/bin/python -m src.v3.prever --canal simulando --auditar                       (Shorts já publicados do canal)
"""
import argparse
import json
import re
import urllib.request
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.v3 import embeddings as emb
from src.v3.base import contexto_canal, feats_descricao, feats_titulo
from src.v3.config import MODELOS, TOKENS, V3
from src.v3.features import CABECAS, KNN_COLS, QUADROS, demanda, lingua_cod, montar_X
from src.v3.quadros import quadros_do_video_local, recorte_central, stats

# Veredito (limiares escolhidos nas previsões fora da amostra; ver reports/v3/validacao.json)
#   VIRAL      nota acima do limiar de maior precisão fora da amostra (alvo_viral no modelo; ~3,5x a taxa base)
#   PROMISSOR  nota acima do limiar de 20% de precisão (2x a taxa base de ~9%)
#   NÃO VIRAL  nota abaixo do limiar em que a taxa de estouro cai para <= 40% da taxa base
#   NEUTRO     o resto
_ART = {}


def art_mult():
    return artefato()["mult_viral"]


def artefato():
    if "m" not in _ART:
        _ART["m"] = joblib.load(MODELOS / "modelo.joblib")
    return _ART["m"]


# ----------------------------------------------------------------------------------------------- canal
def _cliente(canal):
    import socket
    socket.setdefaulttimeout(60)        # sem isso, uma conexão que trava deixa o prever parado para sempre
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    tok = TOKENS / f"{canal}.analise.json"
    if not tok.exists():
        tok = TOKENS / "simulando.analise.json"
    creds = Credentials.from_authorized_user_file(str(tok))
    if not creds.valid:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _dur(s):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return np.nan
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


@lru_cache(maxsize=8)
def historico(canal: str) -> pd.DataFrame:
    """Shorts públicos do canal com data, título, duração e views atuais. `canal` = simulando | pessoal | UC..."""
    yt = _cliente(canal if not canal.startswith("UC") else "simulando")
    cid = canal if canal.startswith("UC") else yt.channels().list(part="id", mine=True).execute(num_retries=3)["items"][0]["id"]
    ids, tok = [], None
    while True:
        r = yt.playlistItems().list(part="contentDetails", playlistId="UU" + cid[2:], maxResults=50, pageToken=tok).execute(num_retries=3)
        ids += [x["contentDetails"]["videoId"] for x in r.get("items", [])]
        tok = r.get("nextPageToken")
        if not tok or len(ids) >= 600:
            break
    linhas = []
    for k in range(0, len(ids), 50):
        r = yt.videos().list(part="snippet,contentDetails,statistics,status", id=",".join(ids[k:k + 50])).execute(num_retries=3)
        for it in r.get("items", []):
            if it["status"].get("privacyStatus") != "public":
                continue
            sn, st = it["snippet"], it.get("statistics", {})
            linhas.append({"id": it["id"], "canal_id": cid, "titulo": sn.get("title"), "descricao": sn.get("description") or "",
                           "tags": sn.get("tags") or [], "lingua": (sn.get("defaultAudioLanguage") or "pt").split("-")[0],
                           "duracao_s": _dur(it["contentDetails"].get("duration")), "publicado": pd.Timestamp(sn["publishedAt"]),
                           "views": float(st.get("viewCount", 0))})
    df = pd.DataFrame(linhas)
    df = df[df["duracao_s"] <= 180].sort_values("publicado").reset_index(drop=True)
    ch = yt.channels().list(part="snippet", id=cid).execute(num_retries=3)["items"][0]
    df.attrs["criado"] = ch["snippet"].get("publishedAt")
    return df


def capa_publicada(vid):
    from PIL import Image
    for nome in ("hqdefault.jpg", "oar2.jpg"):
        try:
            im = Image.open(BytesIO(urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/{nome}", timeout=20).read()))
            return recorte_central(im.convert("RGB"))
        except Exception:  # noqa: BLE001
            continue
    return None


def quadros_publicados(vid):
    from PIL import Image
    ims = []
    for k in (1, 2, 3):
        try:
            d = urllib.request.urlopen(f"https://i.ytimg.com/vi/{vid}/hq{k}.jpg", timeout=20).read()
            ims.append(recorte_central(Image.open(BytesIO(d)).convert("RGB")))
        except Exception:  # noqa: BLE001
            pass
    return ims


@lru_cache(maxsize=4)
def quadros_medios(ids: tuple):
    """Visual médio dos Shorts publicados (vetor CLIP dos quadros + cor, texto na tela...). Usado quando a ideia ainda
    não tem vídeo nem capa: a nota vira "este título, com o visual de sempre do canal"."""
    vets, feats = [], []
    for v in ids:
        ims = quadros_publicados(v)
        if not ims:
            continue
        Vq, aq = emb.emb_imagens(ims)
        m = Vq.mean(0)
        vets.append(m / np.linalg.norm(m))
        feats.append({"qd_animado": float(aq.mean()), **stats(ims, None, Vq)})
    if not vets:
        return None, None
    m = np.mean(vets, 0)
    return m / np.linalg.norm(m), pd.DataFrame(feats).mean().to_dict()


def duracao_video(video):
    import subprocess
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def proximo_horario():
    """Próximo horário de postagem do estúdio: 12:00 ou 18:00 de Brasília (15:00/21:00 UTC)."""
    agora = datetime.now(timezone.utc)
    for d in range(0, 3):
        for h in (15, 21):
            t = (agora + timedelta(days=d)).replace(hour=h, minute=0, second=0, microsecond=0)
            if t > agora + timedelta(minutes=30):
                return t
    return agora + timedelta(hours=6)


# ----------------------------------------------------------------------------------------------- pool da base
@lru_cache(maxsize=1)
def pool():
    b = pd.read_parquet(V3 / "base_v3.parquet", columns=["id", "canal_id", "canal", "titulo", "publicado", "views", "regua",
                                                         "outlier", "log_outlier", "viral", "no_treino"])
    b = b[b["no_treino"] & (b["publicado"] >= pd.Timestamp(artefato()["inicio"], tz="UTC"))].reset_index(drop=True)
    E, tem, _ = emb.carregar("titulos", b["id"].tolist())
    b, E = b[tem].reset_index(drop=True), E[tem]
    return b, E


def arq_auditoria(canal):
    return V3 / f"auditoria_{canal}.json"


def auditoria(canal):
    """Notas que o modelo atual daria aos Shorts já publicados do canal (gerada por --auditar), se existir."""
    f = arq_auditoria(canal)
    if not f.exists():
        return None
    a = json.load(open(f))
    return a if a.get("modelo_mtime") == (MODELOS / "modelo.joblib").stat().st_mtime else None


def _t(s):
    return pd.to_datetime(pd.Series(s), utc=True).dt.tz_convert(None).astype("datetime64[ns]").astype("int64").to_numpy()


# ----------------------------------------------------------------------------------------------- candidato
def candidato_da_entrega(pasta: Path) -> dict:
    from PIL import Image
    post = json.load(open(pasta / "post.json"))
    video = pasta / post["video"] if post.get("video") else None
    capa = pasta / post["capa"] if post.get("capa") else None
    quando = post.get("publicar_em")
    return {"titulo": post["titulo"], "descricao": post.get("descricao", ""), "tags": post.get("tags", []),
            "lingua": (post.get("idioma") or "pt").split("-")[0], "video": str(video) if video and video.exists() else None,
            "capa": Image.open(capa).convert("RGB") if capa and capa.exists() else None,
            "publicado": pd.Timestamp(quando) if quando else pd.Timestamp(proximo_horario()), "canal": post.get("canal", "simulando")}


def _prep_imagem(im):
    return recorte_central(im) if im is not None else None


def avaliar(canal: str, candidatos: list[dict], hist: pd.DataFrame | None = None, vizinhos=5, capas_hist=None):
    """Cada candidato: {titulo, publicado, duracao_s?, video?, capa? (PIL), quadros? (lista PIL), descricao?, tags?}."""
    art = artefato()
    hist = historico(canal) if hist is None else hist
    b, Eb = pool()
    tb = _t(b["publicado"])
    cb = pd.factorize(b["canal_id"])[0]
    proprio = (b["canal_id"] == hist["canal_id"].iloc[0]).to_numpy()
    canal_cod = int(cb[proprio][0]) if proprio.any() else -1
    # histórico: títulos e capas publicadas (para régua, novidade e parecença)
    Eh = emb.emb_titulos(hist["titulo"].tolist())
    capas_hist = capas_hist if capas_hist is not None else [capa_publicada(v) for v in hist["id"]]
    ok_h = np.array([c is not None for c in capas_hist])
    Ch = np.full((len(hist), 512), np.nan, np.float32)
    anim_h = np.full(len(hist), np.nan, np.float32)
    if ok_h.any():
        Ch[ok_h], anim_h[ok_h] = emb.emb_imagens([c for c in capas_hist if c is not None])
    res = []
    for c in candidatos:
        dur = c.get("duracao_s") or (duracao_video(c["video"]) if c.get("video") else None) or 60.0
        novo = {"id": "__candidato__", "canal_id": hist["canal_id"].iloc[0], "titulo": c["titulo"], "descricao": c.get("descricao", ""),
                "tags": c.get("tags", []), "lingua": c.get("lingua", "pt"), "duracao_s": dur,
                "publicado": pd.Timestamp(c["publicado"]).tz_convert("UTC") if pd.Timestamp(c["publicado"]).tzinfo
                else pd.Timestamp(c["publicado"], tz="UTC"), "views": np.nan}
        df = pd.concat([hist.assign(_ord=np.arange(len(hist))), pd.DataFrame([{**novo, "_ord": -1}])], ignore_index=True)
        df["publicado"] = pd.to_datetime(df["publicado"], utc=True)
        df["idade_dias"] = (pd.Timestamp(c["publicado"]) - df["publicado"]).dt.total_seconds() / 86400
        df = contexto_canal(df.sort_values("publicado").reset_index(drop=True),
                            {hist["canal_id"].iloc[0]: {"criado": hist.attrs.get("criado")}})
        df = pd.concat([df, pd.DataFrame([feats_titulo(t) for t in df["titulo"]], index=df.index),
                        pd.DataFrame([feats_descricao(d, tg) for d, tg in zip(df["descricao"], df["tags"])], index=df.index)], axis=1)
        df["pub_hora_utc"] = df["publicado"].dt.hour + df["publicado"].dt.minute / 60
        df["pub_dia_semana"] = df["publicado"].dt.day_of_week
        df["log_outlier"] = np.log(df["outlier"].clip(lower=1e-3))
        df["lingua_cod"] = lingua_cod(df["lingua"])
        n = len(df)
        i = int(np.where(df["id"] == "__candidato__")[0][0])
        pos = df["_ord"].to_numpy()
        Et = np.zeros((n, Eh.shape[1]), np.float32)
        Ec = np.full((n, 512), np.nan, np.float32)
        anim = np.full(n, np.nan, np.float32)
        for r in range(n):
            if r != i:
                Et[r], Ec[r], anim[r] = Eh[pos[r]], Ch[pos[r]], anim_h[pos[r]]
        Et[i] = emb.emb_titulos([c["titulo"]])[0]
        capa = c.get("capa") if c.get("capa") is not None else None
        qims = c.get("quadros") or (quadros_do_video_local(c["video"], dur) if c.get("video") else [])
        visual_medio = False
        if capa is None and qims:
            capa = qims[0]
        if capa is not None:
            V, a = emb.emb_imagens([_prep_imagem(capa)])
            Ec[i], anim[i] = V[0], a[0]
        elif ok_h.any():                                    # só a ideia: capa no estilo médio dos últimos 10 Shorts
            m = Ch[ok_h][-10:].mean(0)
            Ec[i], anim[i] = m / np.linalg.norm(m), float(np.nanmean(anim_h[ok_h][-10:]))
            visual_medio = True
        Q = np.full((n, 512), np.nan, np.float32)
        tem_q = np.zeros(n, bool)
        for col in QUADROS:
            df[col] = np.nan
        if qims:
            Vq, aq = emb.emb_imagens(qims)
            m = Vq.mean(0)
            Q[i] = m / np.linalg.norm(m)
            tem_q[i] = True
            for k, v in {"qd_animado": float(aq.mean()), **stats(qims, None, Vq)}.items():
                df.loc[i, k] = v
        else:                                               # só a ideia: vídeo no estilo médio dos últimos 10 Shorts
            qv, qf = quadros_medios(tuple(hist["id"].tail(10)))
            if qv is not None:
                Q[i], tem_q[i] = qv, True
                for k, v in qf.items():
                    df.loc[i, k] = v
                visual_medio = True
        df["capa_animado"] = anim
        tem_c = ~np.isnan(Ec).any(1)
        X, R, _ = montar_X(df, Et, Ec, tem_c, Q, tem_q, proj=art["proj"])
        # cabeças lineares e demanda
        cab = art["cabecas"]
        h = [cab["titulo"].predict(np.hstack([Et[i], R[i]])[None])[0] if cab.get("titulo") else np.nan,
             cab["capa"].predict(Ec[i][None])[0] if cab.get("capa") is not None and tem_c[i] else np.nan,
             cab["quadros"].predict(Q[i][None])[0] if cab.get("quadros") is not None and tem_q[i] else np.nan]
        tq = _t([c["publicado"]])
        dem = demanda(Et[i][None], tq, np.array([canal_cod]), Eb, tb, cb, b["log_outlier"].to_numpy(float), b["viral"].to_numpy(float))
        linha = pd.concat([X.iloc[[i]].reset_index(drop=True), pd.DataFrame(dem, columns=KNN_COLS),
                           pd.DataFrame([h], columns=CABECAS)], axis=1).reindex(columns=art["colunas"]).astype(np.float32)
        L = linha.to_numpy()
        pb, pr = art["modelo"].predict_proba(L)[:, 1], art["regressao"].predict(L)
        bruto = float(0.5 * (np.searchsorted(art["ref_b"], pb, side="right")[0] / len(art["ref_b"])
                             + np.searchsorted(art["ref_r"], pr, side="right")[0] / len(art["ref_r"])))
        prob = float(art["isotonica"].predict([bruto])[0])
        lim = art["limiares_precisao"]
        av, ap_ = art.get("alvo_viral", "0.3"), art.get("alvo_promissor", "0.2")
        if lim.get(av) is not None and bruto >= lim[av]:
            veredito = "VIRAL"
        elif lim.get(ap_) is not None and bruto >= lim[ap_]:
            veredito = "PROMISSOR"
        elif art["limiar_nao_viral"] is not None and bruto <= art["limiar_nao_viral"]:
            veredito = "NÃO VIRAL"
        else:
            veredito = "NEUTRO"
        # o que puxou: contribuições da regressão (em log do múltiplo da régua: +0,30 = cerca de 1,35x mais views)
        contrib = art["regressao"].booster_.predict(L, pred_contrib=True)[0][:-1]
        top = pd.Series(contrib, index=art["colunas"]).sort_values()
        s = Eb @ Et[i]
        s[tb > tq[0] - 14 * 86400e9] = -9
        if canal_cod >= 0:
            s[cb == canal_cod] = -9
        idx = np.argsort(-s)[:vizinhos]
        viz = [{"titulo": b["titulo"].iloc[j], "canal": b["canal"].iloc[j], "sim": round(float(s[j]), 3),
                "views": int(b["views"].iloc[j]), "x_regua": round(float(b["outlier"].iloc[j]), 1)} for j in idx]
        regua = float(df.loc[i, "regua"]) if pd.notna(df.loc[i, "regua"]) else None
        aud = auditoria(canal)
        no_canal = None
        if aud and aud["linhas"]:
            notas = np.array([x["nota_bruta"] for x in aud["linhas"]])
            melhor = max(aud["linhas"], key=lambda x: x["nota_bruta"])
            no_canal = {"acima_de": int((notas < bruto).sum()), "de": len(notas),
                        "melhor_nota_antes": {"titulo": melhor["titulo"], "views": melhor["views"], "prob": melhor["prob"]}}
        res.append({"titulo": c["titulo"], "prob_estourar": prob, "nota_bruta": bruto, "veredito": veredito,
                    "multiplo_previsto": float(np.exp(pr[0])),
                    "percentil_na_base": float((art["oof"]["p"] < bruto).mean()), "regua_canal": regua, "no_canal": no_canal,
                    "meta_views": max(round(regua * art["mult_viral"]), art.get("views_min_viral", 0)) if regua else None, "taxa_base": art["taxa_base"],
                    "com_video": bool(tem_q[i]) and not visual_medio, "com_capa": bool(tem_c[i]) and not visual_medio,
                    "visual_medio_do_canal": visual_medio,
                    "puxa_para_cima": top.tail(6)[::-1].round(3).to_dict(), "puxa_para_baixo": top.head(6).round(3).to_dict(),
                    "parecidos_outros_canais": viz})
    return res


def imprimir(res):
    for r in res:
        print("=" * 104)
        print(r["titulo"])
        meta = f"{r['meta_views']:,} views" if r["meta_views"] else "?"
        print(f"  VEREDITO: {r['veredito']}   chance de estourar (>= 5x a régua do canal e >= 5 mil = {meta}): {r['prob_estourar']:.1%}"
              f"   [média {r['taxa_base']:.1%}; nota acima de {r['percentil_na_base']:.0%} da base]"
              f"{'   (sem vídeo nem capa: supus o visual médio dos seus últimos Shorts)' if r.get('visual_medio_do_canal') else '' if r['com_video'] else '   (sem vídeo: só título/capa, menos confiável)'}")
        if r.get("no_canal"):
            nc = r["no_canal"]
            print(f"  entre os seus Shorts já publicados: nota maior que {nc['acima_de']} de {nc['de']}"
                  f" (o de nota mais alta foi \"{nc['melhor_nota_antes']['titulo'][:50]}\", {nc['melhor_nota_antes']['views']:,} views)")
        if r.get("regua_canal"):
            print(f"  views esperadas: ~{r['multiplo_previsto'] * r['regua_canal']:,.0f} ({r['multiplo_previsto']:.2f}x a régua de "
                  f"{r['regua_canal']:,.0f}; é a mediana prevista, não o teto)")
        print("  puxa para cima: ", ", ".join(f"{k} x{np.exp(v):.2f}" for k, v in r["puxa_para_cima"].items() if v > 0))
        print("  puxa para baixo:", ", ".join(f"{k} x{np.exp(v):.2f}" for k, v in r["puxa_para_baixo"].items() if v < 0))
        print("  temas parecidos em outros canais (similaridade, x régua, views):")
        for v in r["parecidos_outros_canais"]:
            print(f"    {v['sim']:.2f}  {v['x_regua']:>6.1f}x  {v['views']:>11,}  {v['canal'][:22]:22s} {v['titulo'][:66]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entrega", action="append", help="pasta com post.json (pode repetir)")
    ap.add_argument("--canal", default="simulando")
    ap.add_argument("--titulo", action="append")
    ap.add_argument("--video")
    ap.add_argument("--capa")
    ap.add_argument("--duracao", type=float)
    ap.add_argument("--quando", help="ISO UTC; padrão: publicar_em do post.json ou o próximo 12:00/18:00 de Brasília")
    ap.add_argument("--descricao", default="")
    ap.add_argument("--auditar", action="store_true", help="aplica o modelo aos Shorts já publicados do canal")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.auditar:
        h = historico(a.canal)
        capas = [capa_publicada(v) for v in h["id"]]
        linhas = []
        for i in range(10, len(h)):
            alvo = h.iloc[i]
            r = avaliar(a.canal, [{"titulo": alvo["titulo"], "duracao_s": alvo["duracao_s"], "publicado": alvo["publicado"],
                                   "capa": capas[i], "quadros": quadros_publicados(alvo["id"]), "descricao": alvo["descricao"],
                                   "tags": alvo["tags"], "lingua": alvo["lingua"]}],
                        hist=h.iloc[:i].reset_index(drop=True), vizinhos=3, capas_hist=capas[:i])[0]
            regua = r["regua_canal"]
            linhas.append({"publicado": str(alvo["publicado"])[:16], "titulo": alvo["titulo"][:55], "views": int(alvo["views"]),
                           "x_regua": round(alvo["views"] / regua, 2) if regua else None, "prob": round(r["prob_estourar"], 3),
                           "nota_bruta": r["nota_bruta"], "veredito": r["veredito"]})
        json.dump({"canal": a.canal, "modelo_mtime": (MODELOS / "modelo.joblib").stat().st_mtime, "linhas": linhas},
                  open(arq_auditoria(a.canal), "w"), ensure_ascii=False, indent=1, default=float)
        t = pd.DataFrame(linhas).drop(columns="nota_bruta")
        print(t.to_string(index=False))
        estourou = (t["x_regua"].fillna(0) >= art_mult()) & (t["views"] >= artefato().get("views_min_viral", 0))
        print(f"\n{len(t)} Shorts avaliados; vereditos: {t['veredito'].value_counts().to_dict()}; "
              f"estouraram de fato: {int(sum(estourou))}")
    else:
        from PIL import Image
        cands = []
        for pasta in a.entrega or []:
            c = candidato_da_entrega(Path(pasta))
            if a.quando:
                c["publicado"] = pd.Timestamp(a.quando)
            a.canal = c.pop("canal")
            cands.append(c)
        if a.titulo:
            capa = Image.open(a.capa).convert("RGB") if a.capa else None
            quando = pd.Timestamp(a.quando) if a.quando else pd.Timestamp(proximo_horario())
            cands += [{"titulo": t, "duracao_s": a.duracao, "video": a.video, "publicado": quando, "capa": capa,
                       "descricao": a.descricao} for t in a.titulo]
        res = avaliar(a.canal, cands)
        if a.json:
            print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
        else:
            imprimir(sorted(res, key=lambda r: -r["prob_estourar"]))
