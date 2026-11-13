"""Teste de features novas para o modelo v2, todas tiradas de dados que já estão no disco (nada é baixado de novo).

fala        da legenda original, com o tempo de cada palavra: ritmo em letras por segundo, pausas, quanto o vídeo
            continua depois da última palavra, variação do ritmo e se os primeiros 5 s repetem o título.
publicação  do info.json do yt-dlp, decididas antes de publicar: hora e dia, dublagem (número de faixas de áudio),
            resolução e quadros por segundo.
atipicidade a ideia do Bravin et al. (2026): quanto o vídeo foge do padrão do próprio canal. Para cada grupo, é a
            distância entre o vídeo e a média do canal, com as features padronizadas.
controle    não é decisão do criador: a idade do vídeo no dia da coleta. Serve para ver se o alvo é justo, não
            para prever um vídeo novo (na hora de publicar, a idade é zero para todos).

Comparação pareada: as mesmas divisões por canal, repetidas com sementes diferentes, com e sem as features novas.
Como em v2_modelo.py, cada feature entra relativa à média do próprio canal.

Resultado (13/11/2026, 5 divisões por canal, diferença pareada contra as 49 features):
- fala e publicação não ajudam: no Random Forest, de -0,006 a -0,028; nenhuma tem |rho| > 0,11 com o percentil;
- a atipicidade ajuda só a Logística: metadados + cenas + atipicidade = 0,601 (+0,042, melhor em 5 de 5),
  empatando com o Random Forest com tudo (0,599). O sinal é o oposto do Bravin: o vídeo que foge do padrão do
  canal vai um pouco pior (rho = -0,10, p = 0,06). Lá a medida era engajamento numa tendência; aqui é alcance.

Uso: .venv/bin/python -I src/v2/v2_features_novas.py <pasta v2> <pasta videos/>   (depois de rodar v2_modelo.py)
"""
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v2_modelo as M  # noqa: E402  (lê <pasta v2> e <pasta videos/> de sys.argv)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402

REPETICOES = int(os.environ.get("REPETICOES", 5))


def f_fala(vid, dur, titulo, lingua):
    arq = M.legenda_original(vid, lingua)
    if not arq or not dur:
        return {}
    pal = []
    for ev in json.load(open(arq[0])).get("events", []):
        for s in ev.get("segs") or []:
            w = s.get("utf8", "").strip()
            if w and not w.startswith("["):
                pal.append(((ev.get("tStartMs", 0) + s.get("tOffsetMs", 0)) / 1000, w))
    if len(pal) < 5:
        return {}
    pal.sort()
    t = np.array([x for x, _ in pal])
    gaps = np.diff(t)                                               # de um início de palavra ao seguinte
    por_5s = np.histogram(t, bins=np.arange(0, t[-1] + 5, 5))[0]    # palavras em cada janela de 5 s, até a última
    tit = {w for w in re.findall(r"\w+", (titulo or "").lower()) if len(w) >= 4}
    inicio = {w for x, p in pal if x < 5 for w in re.findall(r"\w+", p.lower())}
    return {"fala_letras_s": sum(len(re.sub(r"\W", "", p)) for _, p in pal) / dur,
            "fala_pausa_max_s": float(gaps.max()),
            "fala_pausas_min": float((gaps > 1.0).sum() / (dur / 60)),
            "fala_fim_s": max(0.0, dur - t[-1]),
            "fala_ritmo_cv": float(por_5s.std() / por_5s.mean()) if por_5s.mean() > 0 else np.nan,
            "fala_titulo_inicio": len(tit & inicio) / len(tit) if tit else np.nan}


def f_publicacao(vid):
    info = M.B / "videos" / f"{vid}.info.json"
    if not info.exists():
        return {}
    j = json.load(open(info))
    ts = j.get("timestamp")
    formatos = j.get("formats") or []
    audios = {f.get("language") for f in formatos if f.get("acodec") not in (None, "none") and f.get("language")}
    out = {"pub_faixas_audio": max(1, len(audios)),
           "pub_altura_max": max([f.get("height") or 0 for f in formatos] or [np.nan]),
           "pub_fps_max": max([f.get("fps") or 0 for f in formatos] or [np.nan])}
    if ts:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        out.update({"pub_hora_utc": dt.hour + dt.minute / 60, "pub_fim_semana": int(dt.weekday() >= 5),
                    "ctl_idade_dias": (j.get("epoch", ts) - ts) / 86400})
    return out


def dobras(g, semente, k=5):
    canais = np.unique(g)
    np.random.default_rng(semente).shuffle(canais)
    dobra = dict(zip(canais, np.arange(len(canais)) % k))
    idx = np.array([dobra[c] for c in g])
    return [(np.where(idx != i)[0], np.where(idx == i)[0]) for i in range(k)]


def auc_aninhada(X, y, g, nome, divisoes):
    est, grade = M.MODELOS[nome]
    aucs = []
    for tr, te in divisoes:
        busca = GridSearchCV(M.pipe(est), grade, scoring="roc_auc", cv=GroupKFold(n_splits=4), n_jobs=-1)
        busca.fit(X.iloc[tr], y[tr], groups=g[tr])
        aucs.append(roc_auc_score(y[te], busca.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs))


if __name__ == "__main__":
    base = pd.read_csv(M.S / "base_v2.csv")
    novas = pd.DataFrame([{**f_fala(r.id, r.duracao_s, r.titulo, r.lingua), **f_publicacao(r.id)}
                          for r in base.itertuples()], index=base.index)
    base = pd.concat([base, novas], axis=1)
    y = (base["percentil"] >= 0.8).astype(int).values
    g = base["canal"].values
    atuais = [c for cols in M.GRUPOS.values() for c in cols]
    fala = [c for c in novas.columns if c.startswith("fala_")]
    pub = [c for c in novas.columns if c.startswith("pub_")]
    ctl = [c for c in novas.columns if c.startswith("ctl_")]
    X = M.relativo(base, atuais + fala + pub + ctl)
    Z = X[atuais] / X[atuais].std()                      # desvio de cada feature para a média do canal, em desvios-padrão
    for grupo, cols in M.GRUPOS.items():
        X[f"atip_{grupo}"] = np.sqrt((Z[cols] ** 2).mean(axis=1))
    X["atip_tudo"] = np.sqrt((Z ** 2).mean(axis=1))
    atip = [c for c in X.columns if c.startswith("atip_")]

    print(f"Base: {len(base)} Shorts, {base['canal'].nunique()} canais; línguas: {base['lingua'].value_counts().to_dict()}")
    print(f"Com features de fala: {base['fala_letras_s'].notna().sum()}; vídeos dublados (mais de 1 faixa): "
          f"{(base['pub_faixas_audio'] > 1).sum()} em {base.loc[base['pub_faixas_audio'] > 1, 'canal'].nunique()} canais")

    print("\n== Cada feature nova sozinha: Spearman (feature relativa ao canal x percentil no canal) ==")
    linhas = []
    for c in fala + pub + ctl + atip:
        ok = X[c].notna()
        if X.loc[ok, c].nunique() < 3:
            print(f"{c:22s} quase sem variação dentro do canal (n = {ok.sum()})")
            continue
        rho, p = spearmanr(X.loc[ok, c], base.loc[ok, "percentil"])
        linhas.append({"feature": c, "rho": round(rho, 3), "p": round(p, 4), "n": int(ok.sum())})
    print(pd.DataFrame(linhas).sort_values("rho", key=abs, ascending=False).to_string(index=False))

    conjuntos = {"atual (49 features)": atuais, "+ fala": atuais + fala, "+ publicação": atuais + pub,
                 "+ fala + publicação": atuais + fala + pub, "+ controle de idade": atuais + ctl,
                 "metadados + cenas": M.GRUPOS["metadados"] + M.GRUPOS["cenas"],
                 "metadados + cenas + fala + publicação": M.GRUPOS["metadados"] + M.GRUPOS["cenas"] + fala + pub,
                 "+ atipicidade": atuais + atip, "metadados + cenas + atipicidade": M.GRUPOS["metadados"] + M.GRUPOS["cenas"] + atip}
    if os.environ.get("SO"):
        conjuntos = {k: v for k, v in conjuntos.items() if k in os.environ["SO"].split(",") or k == "atual (49 features)"}
    print(f"\n== AUC na validação aninhada por canal, {REPETICOES} divisões diferentes (média ± dp entre divisões) ==")
    res = {}
    for nome in ["Random Forest", "Regressão Logística"]:
        for rotulo, cols in conjuntos.items():
            res[(nome, rotulo)] = [auc_aninhada(X[cols], y, g, nome, dobras(g, s)) for s in range(REPETICOES)]
            a = np.array(res[(nome, rotulo)])
            ref = np.array(res[(nome, "atual (49 features)")])
            dif = a - ref
            print(f"{nome:20s} {rotulo:40s} AUC {a.mean():.3f} ± {a.std():.3f}   diferença pareada {dif.mean():+.3f} "
                  f"(melhorou em {(dif > 0).sum()} de {len(dif)})", flush=True)
    json.dump({"spearman": linhas, "auc": {f"{n} | {r}": v for (n, r), v in res.items()}},
              open(M.S / "features_novas.json", "w"), ensure_ascii=False, indent=1)
    pd.concat([base[["id", "canal"] + fala + pub + ctl], X[atip]], axis=1).to_csv(M.S / "features_novas.csv", index=False)
