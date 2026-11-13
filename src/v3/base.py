"""Monta a base v3: uma linha por Short, com o rótulo e as features que existem ANTES de postar.

Rótulo (pergunta do criador: "este Short vai estourar no MEU canal?")
    régua    = mediana das views dos JANELA_REGUA Shorts anteriores do mesmo canal (só o passado)
    outlier  = views / régua
    viral    = outlier >= MULT_VIRAL (5x) e views >= VIEWS_MIN_VIRAL (5 mil)
Assim o tamanho do canal sai da conta: 5x num canal de 1.500 views é 7.500; num de 100 mil, 500 mil. O piso de 5 mil
evita chamar de viral um Short que foi de 120 para 1.800 views; canais com régua < 300 ficam fora do treino.

Features de contexto do canal (todas só com Shorts publicados antes): régua, dispersão, taxa de estouros anteriores,
maior estouro anterior, ritmo de postagem, tendência recente, duração relativa e idade do canal.
"""
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

from src.v3.config import (ABAS, CANAL_IDADE_MIN_TREINO, IDADE_MIN_DIAS, JANELA_REGUA, MATURO_DIAS, MIN_ANTERIORES,
                           MULT_VIRAL, N_ANTERIORES_MIN_TREINO, REGUA_MIN, V3, VIEWS_MIN_VIRAL)

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F000-\U0001F2FF]")
HIPOTESE = re.compile(r"\b(e se|what if|y si|et si|was wäre wenn|und wenn|o que aconteceria|o que acontece(ria)? se|"
                      r"what would happen|what happens (if|when)|qué pasaría|que pasaria|qué pasa si|que se passerait|"
                      r"was passiert,? wenn|imagine|imagina)\b", re.I)
COMO_PORQUE = re.compile(r"\b(como|how|cómo|comment|wie|por ?que|porque|why|por qué|pourquoi|warum)\b", re.I)
VOCE = re.compile(r"\b(você|voce|vc|seu|sua|you|your|tú|tu|usted|tus|vous|votre|du|dich|dein)\b", re.I)
EXTREMO = re.compile(r"\b(mais|maior|menor|pior|melhor|nunca|ningu[eé]m|imposs[ií]vel|incr[ií]vel|mundo|most|biggest|"
                     r"largest|worst|best|never|nobody|impossible|insane|world'?s?|más|mayor|peor|mejor|nunca|nadie|"
                     r"imposible|plus|jamais|größte|nie|unmöglich)\b", re.I)
TEMPO = re.compile(r"\b(\d+\s*(segundos?|minutos?|horas?|dias?|anos?|seconds?|minutes?|hours?|days?|years?|"
                   r"segundos?|años|secondes?|ans|sekunden|jahre))\b", re.I)
PERIGO = re.compile(r"\b(morte|morrer|morreria|explod|destr|fim do mundo|apocalipse|death|die|dies|explod|destroy|"
                    r"end of the world|muerte|morir|destru|mort|tod|sterben|zerst)\w*", re.I)


def _dur(s):
    if not s:
        return np.nan
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return np.nan
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def feats_titulo(t: str) -> dict:
    t = t or ""
    limpo = re.sub(r"#\w+", "", t).strip()
    letras = [c for c in limpo if c.isalpha()]
    return {
        "tit_caracteres": len(limpo), "tit_palavras": len(limpo.split()),
        "tit_interrogacao": int("?" in t), "tit_exclamacao": int("!" in t), "tit_reticencias": int("..." in t or "…" in t),
        "tit_numeros": len(re.findall(r"\d+", limpo)), "tit_emojis": len(EMOJI.findall(t)),
        "tit_maiusculas": sum(c.isupper() for c in letras) / max(1, len(letras)),
        "tit_hashtags": len(re.findall(r"#\w+", t)), "tit_tem_shorts": int(bool(re.search(r"#shorts?\b", t, re.I))),
        "tit_hipotese": int(bool(HIPOTESE.search(t))), "tit_como_porque": int(bool(COMO_PORQUE.search(t))),
        "tit_voce": int(bool(VOCE.search(t))), "tit_extremo": int(bool(EXTREMO.search(t))),
        "tit_tempo": int(bool(TEMPO.search(t))), "tit_perigo": int(bool(PERIGO.search(t))),
    }


def feats_descricao(d: str, tags) -> dict:
    d = d or ""
    return {"desc_caracteres": len(d), "desc_hashtags": len(re.findall(r"#\w+", d)),
            "desc_links": len(re.findall(r"https?://", d)), "n_tags": len(tags or [])}


def _lingua(r):
    for k in ("lingua_audio", "lingua"):
        v = (r.get(k) or "").split("-")[0].lower()
        if v and v != "zxx":
            return v
    return "?"


def carregar_brutos() -> pd.DataFrame:
    """Junta abas (lista do canal) e Data API (data, duração, língua...). Fica só o que a API devolveu."""
    api = {}
    for l in open(V3 / "videos_api.jsonl"):
        r = json.loads(l)
        if not r.get("indisponivel"):
            api[r["id"]] = r
    linhas = []
    for arq in sorted(ABAS.glob("*.jsonl")):
        for l in open(arq):
            a = json.loads(l)
            r = api.get(a["id"])
            if not r:
                continue
            linhas.append({
                "id": r["id"], "canal_id": r["canal_id"], "canal": r.get("canal"), "titulo": r.get("titulo") or a.get("titulo"),
                "descricao": r.get("descricao") or "", "tags": r.get("tags") or [], "categoria": r.get("categoria"),
                "lingua": _lingua(r), "duracao_s": _dur(r.get("duracao")),
                "publicado": pd.Timestamp(r["publicado"]), "coletado": pd.Timestamp(r["coletado"]),
                "views": r.get("views"), "likes": r.get("likes"), "comentarios": r.get("comentarios"),
            })
    df = pd.DataFrame(linhas).drop_duplicates("id")
    df = df[df["views"].notna() & (df["duracao_s"] <= 180) & (df["duracao_s"] > 0)]
    df["idade_dias"] = (df["coletado"] - df["publicado"]).dt.total_seconds() / 86400
    return df.sort_values(["canal_id", "publicado"]).reset_index(drop=True)


def contexto_canal(df: pd.DataFrame, canais_api: dict | None = None, maturo_dias: float = MATURO_DIAS) -> pd.DataFrame:
    """Régua, rótulo e features do canal, todas com Shorts publicados ANTES de cada um.

    A régua do rótulo usa os 30 Shorts anteriores (definição do alvo). As FEATURES que dependem de views só usam Shorts
    com >= `maturo_dias` na hora de postar: no treino as views são as finais, e na vida real um Short de 2 dias ainda
    não mostrou o que vai fazer — sem esse corte o modelo aprende com a onda que o canal ainda nem sabia que vinha."""
    canais_api = canais_api or {}
    saida = []
    for cid, g in df.groupby("canal_id", sort=False):
        g = g.sort_values("publicado")
        v = g["views"].to_numpy(float)
        lv = np.log1p(v)
        t = g["publicado"].to_numpy()
        dur = g["duracao_s"].to_numpy(float)
        criado = canais_api.get(cid, {}).get("criado")
        criado = pd.Timestamp(criado) if criado else None
        outl = np.full(len(g), np.nan)
        reg = np.full(len(g), np.nan)
        for i in range(len(g)):
            ant = v[max(0, i - JANELA_REGUA):i]
            if len(ant) >= MIN_ANTERIORES:
                reg[i] = np.median(ant)
                outl[i] = v[i] / max(reg[i], 1.0)
        for i in range(len(g)):
            a0 = max(0, i - JANELA_REGUA)
            dias_ant = (t[i] - t[:i]) / np.timedelta64(1, "D") if i else np.array([])
            maduros = np.where(dias_ant >= maturo_dias)[0] if i else np.array([], int)
            if len(maduros) < MIN_ANTERIORES and i:      # canal novo/postando em rajada: usa o que já tem >= 2 dias
                maduros = np.where(dias_ant >= 2)[0]
            m30 = maduros[-JANELA_REGUA:]
            ant_lv = lv[m30]
            reg_m = float(np.median(v[m30])) if len(m30) >= MIN_ANTERIORES else np.nan
            ant_out = outl[maduros[-60:]] if len(maduros) else np.array([])
            ant_out = ant_out[~np.isnan(ant_out)]
            saida.append({
                "id": g["id"].iloc[i],
                "regua": reg[i], "outlier": outl[i],
                "ctx_n_anteriores": i,
                "ctx_log_regua": np.log1p(reg_m) if not np.isnan(reg_m) else np.nan,
                "ctx_dispersao": float(np.std(ant_lv)) if len(ant_lv) >= 3 else np.nan,
                "ctx_p90_rel": float(np.quantile(ant_lv, 0.9) - np.median(ant_lv)) if len(ant_lv) >= 5 else np.nan,
                "ctx_taxa_estouro": float((ant_out >= MULT_VIRAL).mean()) if len(ant_out) >= 5 else np.nan,
                "ctx_taxa_2x": float((ant_out >= 2).mean()) if len(ant_out) >= 5 else np.nan,
                "ctx_max_outlier": float(np.log(ant_out.max())) if len(ant_out) else np.nan,
                "ctx_ultimo_outlier": float(np.log(max(ant_out[-1], 1e-3))) if len(ant_out) else np.nan,
                "ctx_tendencia": float(np.median(lv[m30[-10:]]) - np.median(ant_lv)) if len(m30) >= 10 else np.nan,
                "ctx_dias_desde_ultimo": float(dias_ant[-1]) if i else np.nan,
                "ctx_shorts_30d": int((dias_ant <= 30).sum()) if i else 0,
                "ctx_dur_rel": float(dur[i] / np.median(dur[a0:i])) if i >= 3 else np.nan,
                "ctx_idade_canal_dias": float((g["publicado"].iloc[i] - criado).days) if criado is not None else np.nan,
            })
    return df.merge(pd.DataFrame(saida), on="id", how="left")


def montar(salvar: bool = True) -> pd.DataFrame:
    df = carregar_brutos()
    canais_api = json.load(open(V3 / "canais_api.json")) if (V3 / "canais_api.json").exists() else {}
    df = contexto_canal(df, canais_api)
    df = pd.concat([df, pd.DataFrame([feats_titulo(t) for t in df["titulo"]], index=df.index),
                    pd.DataFrame([feats_descricao(d, tg) for d, tg in zip(df["descricao"], df["tags"])], index=df.index)], axis=1)
    df["pub_hora_utc"] = df["publicado"].dt.hour + df["publicado"].dt.minute / 60
    df["pub_dia_semana"] = df["publicado"].dt.day_of_week
    df["log_outlier"] = np.log(df["outlier"].clip(lower=1e-3))
    df["viral"] = ((df["outlier"] >= MULT_VIRAL) & (df["views"] >= VIEWS_MIN_VIRAL)).astype(float).where(df["outlier"].notna())
    df["no_treino"] = (df["outlier"].notna() & (df["idade_dias"] >= IDADE_MIN_DIAS) & (df["regua"] >= REGUA_MIN)
                       & (df["ctx_n_anteriores"] >= N_ANTERIORES_MIN_TREINO)
                       & ~(df["ctx_idade_canal_dias"] < CANAL_IDADE_MIN_TREINO))
    if salvar:
        df.drop(columns=["descricao"]).to_parquet(V3 / "base_v3.parquet", index=False)
    return df


if __name__ == "__main__":
    t0 = datetime.now()
    df = montar()
    tr = df[df["no_treino"]]
    print(f"{len(df)} Shorts de {df['canal_id'].nunique()} canais; no treino (com régua e >= 14 dias): {len(tr)} "
          f"de {tr['canal_id'].nunique()} canais; viral (>= 5x a régua e >= 5 mil views): {tr["viral"].mean():.1%}  [{datetime.now() - t0}]")
    print(tr["outlier"].describe(percentiles=[.5, .8, .9, .95, .99]).round(2).to_string())
    print(tr["lingua"].value_counts().head(10).to_string())
