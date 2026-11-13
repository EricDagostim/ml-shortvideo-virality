"""Modelo v2: capa, título, roteiro falado e cenas, treinado nos 384 Shorts de canais faceless e testado no Simulando.

Alvo: o Short está no topo do próprio canal (percentil das views na aba Shorts >= 0,8)? Isso tira o tamanho do
canal e o idioma da conta e pergunta o que importa para o criador: qual dos meus vídeos vai melhor.
Avaliação: validação cruzada aninhada com GroupKFold por canal (nenhum canal aparece no treino e no teste ao mesmo tempo).

Uso: .venv/bin/python -I v2_modelo.py <pasta v2> <pasta videos/>
"""
import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
S, V = Path(sys.argv[1]), Path(sys.argv[2])
B = V / "recursos" / "snapugc" / "base"
SEMENTE = 42
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
HIPOTESE = re.compile(r"^\W*(e se|what if|y si|et si|was wäre|und wenn|und was|ماذا لو)", re.I)
PERGUNTA = re.compile(r"^\W*(por ?que|porque|why|por qué|pourquoi|warum|como|how|cómo|comment|wie|o que|what|qué|que se passa)", re.I)
_VOCE = {  # pronomes de 2ª pessoa por língua (o "du" francês é "do", não "você"; por isso a regra depende da língua)
    "pt": "você|voce|vc|vocês|voces|tu|te|teu|tua|teus|tuas|contigo",
    "en": "you|your|yours|yourself|you're|youre",
    "es": "tú|tu|tus|usted|ustedes|te|ti|contigo|vos",
    "de": "du|dich|dir|dein|deine|deinen|deinem|deiner|ihr|euch|euer",
    "fr": "tu|toi|ton|ta|tes|vous|votre|vos|te",
}
VOCE = {k: re.compile(rf"\b({v})\b", re.I) for k, v in _VOCE.items()}
VOCE[""] = re.compile(r"\b(" + "|".join(sorted({w for v in _VOCE.values() for w in v.split("|")} - {"du", "ta", "te"})) + r")\b", re.I)


# ---------- features ----------
def f_titulo(t, lingua=""):
    t = t or ""
    voce = VOCE.get(lingua, VOCE[""])
    return {"tit_caracteres": len(t), "tit_palavras": len(t.split()), "tit_interrogacao": int("?" in t),
            "tit_exclamacao": int("!" in t), "tit_reticencias": int("…" in t or "..." in t),
            "tit_numeros": len(re.findall(r"\d+", t)), "tit_maiusculas": sum(c.isupper() for c in t) / max(1, len(t)),
            "tit_emojis": len(EMOJI.findall(t)), "tit_hipotese": int(bool(HIPOTESE.search(t))),
            "tit_pergunta": int(bool(PERGUNTA.search(t))), "tit_voce": int(bool(voce.search(t))),
            "tit_hashtags": t.count("#")}


def f_descricao(d):
    d = d or ""
    return {"desc_caracteres": len(d), "desc_hashtags": len(re.findall(r"#\w+", d)),
            "desc_links": len(re.findall(r"https?://", d)), "desc_emojis": len(EMOJI.findall(d))}


def _sem_resposta():
    """Ids em que o YouTube deu erro (bloqueio, privado, removido): não dá para saber se têm fala -> NaN, não 0."""
    ids = set()
    for log in ["legendas.log", "legendas2.log"]:                # vale a última tentativa de cada vídeo
        if (S / log).exists():
            texto = open(S / log).read()
            ids -= set(re.findall(r"\[youtube\] ([A-Za-z0-9_-]{11}): Downloading webpage", texto))
            ids |= set(re.findall(r"ERROR: \[youtube\] ([A-Za-z0-9_-]{11}):", texto))
    return ids


SEM_RESPOSTA = _sem_resposta()


def lingua_original(vid):
    """Língua do áudio original, como o yt-dlp registrou no info.json (ex.: "es-US" -> "es")."""
    info = B / "videos" / f"{vid}.info.json"
    return ((json.load(open(info)).get("language") or "") if info.exists() else "").split("-")[0].lower()


def legenda_original(vid, lingua):
    """A legenda automática na língua do áudio original. Vídeo com dublagem tem uma faixa de áudio por língua e uma
    legenda "-orig" para cada faixa; ler outra que não a original é ler a fala dublada, ou seja, uma tradução."""
    arq = sorted(glob.glob(str(S / "legendas" / f"{vid}.*.json3"))) or sorted(glob.glob(str(S / "legendas_whisper" / f"{vid}.*.json3")))
    orig = [a for a in arq if Path(a).name.split(".")[1].split("-")[0] == lingua]
    return (orig or arq)[:1]


def f_roteiro(vid, dur, lingua=""):
    # legenda automática do YouTube, sem tradução; para o Simulando, se faltar, a transcrição local do Whisper
    arq = legenda_original(vid, lingua)
    voce = VOCE.get(lingua, VOCE[""])
    if not dur:
        return {"rot_tem_legenda": 0}
    if not arq:
        return {"rot_tem_legenda": np.nan} if vid in SEM_RESPOSTA else {"rot_tem_legenda": 0}
    palavras, fala_ms = [], 0
    for ev in json.load(open(arq[0])).get("events", []):
        segs = ev.get("segs") or []
        texto = [s for s in segs if s.get("utf8", "").strip() and not s["utf8"].strip().startswith("[")]
        if texto:
            fala_ms += ev.get("dDurationMs", 0)
        for s in texto:
            palavras.append((ev.get("tStartMs", 0) + s.get("tOffsetMs", 0), s["utf8"].strip()))
    if not palavras:
        return {"rot_tem_legenda": 1, "rot_palavras": 0}
    tokens = [p for _, p in palavras]
    n = len(tokens)
    return {"rot_tem_legenda": 1, "rot_palavras": n, "rot_palavras_min": n / (dur / 60),
            "rot_inicio_s": palavras[0][0] / 1000, "rot_palavras_3s": sum(1 for t, _ in palavras if t < 3000),
            "rot_fala_frac": min(1.0, fala_ms / (dur * 1000)), "rot_numeros": sum(any(c.isdigit() for c in p) for p in tokens),
            "rot_voce_100": 100 * sum(bool(voce.fullmatch(p.lower().strip(".,!?"))) for p in tokens) / n,
            "rot_variedade": len({p.lower() for p in tokens}) / n, "rot_tam_palavra": np.mean([len(p) for p in tokens])}


def f_cenas(r):
    sons = (r.get("sons_yamnet") or "").lower()
    return {"cen_ecr": r.get("ecr"), "cen_musica": int("music" in sons), "cen_voz_sintetica": int("synthesizer" in sons),
            "cen_narracao": int("narration" in sons)}


GRUPOS = {
    "metadados": ["duracao_s", "desc_caracteres", "desc_hashtags", "desc_links", "desc_emojis", "n_tags"],
    "título": list(f_titulo("").keys()),
    "capa": [c for c in pd.read_csv(S / "capas.csv").columns if c.startswith("capa_")],
    "roteiro": ["rot_tem_legenda", "rot_palavras", "rot_palavras_min", "rot_inicio_s", "rot_palavras_3s", "rot_fala_frac",
                "rot_numeros", "rot_voce_100", "rot_variedade", "rot_tam_palavra"],
    "cenas": ["cen_cenas", "cen_cenas_min", "cen_plano_medio_s", "cen_distintas", "cen_mudanca_media",
              "cen_ecr", "cen_musica", "cen_voz_sintetica", "cen_narracao"],
}


def juntar_cenas(tabela):
    arq = S / "cenas.csv"
    if not arq.exists():
        return tabela
    cenas = pd.read_csv(arq).drop(columns=["duracao_medida"], errors="ignore").drop_duplicates("id")
    return tabela.merge(cenas, on="id", how="left")


def percentis_canal():
    out = {}
    for f in glob.glob(str(B / "tabs" / "*.jsonl")):
        v = [(x["id"], x["view_count"]) for x in map(json.loads, filter(str.strip, open(f))) if x.get("view_count") is not None]
        if len(v) < 12:
            continue
        ordem = sorted(c for _, c in v)
        for i, c in v:
            out[i] = (sum(o < c for o in ordem) + 0.5 * sum(o == c for o in ordem)) / len(ordem)
    return out


def montar_base():
    pc = percentis_canal()
    capas = pd.read_csv(S / "capas.csv").set_index("id")
    linhas = []
    for r in json.load(open(B / "resultados.json")):
        if r["id"] not in pc or r.get("ecr") is None:
            continue
        info = B / "videos" / f"{r['id']}.info.json"
        tags = len(json.load(open(info)).get("tags") or []) if info.exists() else np.nan
        dur, lingua = r.get("duracao"), lingua_original(r["id"])
        linha = {"id": r["id"], "canal": r["canal"], "titulo": r.get("titulo"), "views": r.get("views"),
                 "percentil": pc[r["id"]], "lingua": lingua, "duracao_s": dur, "n_tags": tags,
                 **f_titulo(r.get("titulo"), lingua), **f_descricao(r.get("descricao")), **f_roteiro(r["id"], dur, lingua),
                 **f_cenas(r)}
        if r["id"] in capas.index:
            linha.update(capas.loc[r["id"]].to_dict())
        linhas.append(linha)
    return juntar_cenas(pd.DataFrame(linhas))


def montar_simulando():
    reg = json.load(open(V / "estudio-shorts" / "analise" / "dados" / "videos.json"))["videos"]
    ecr = {x["id"]: x for x in json.load(open(B / "meus_resultados.json"))}
    if (S / "ecr_simulando.json").exists():
        ecr.update({x["id"]: x for x in json.load(open(S / "ecr_simulando.json"))})
    canal_csv = pd.read_csv(sys.argv[3]) if len(sys.argv) > 3 else None   # data/canal/simulando_o_mundo.csv (views)
    capas = pd.read_csv(S / "capas.csv").set_index("id")
    linhas = []
    for vid, f in reg.items():
        if f.get("canal") != "simulando" or f.get("privacidade") != "public":
            continue
        dur = f.get("duracao_s")
        linha = {"id": vid, "canal": "Simulando o Mundo", "titulo": f.get("titulo"), "duracao_s": dur,
                 "n_tags": len(f.get("tags") or []), **f_titulo(f.get("titulo"), "pt"), **f_descricao(f.get("descricao")),
                 **f_roteiro(vid, dur, "pt"), **f_cenas(ecr.get(vid, {}))}
        if vid in capas.index:
            linha.update(capas.loc[vid].to_dict())
        linhas.append(linha)
    sim = juntar_cenas(pd.DataFrame(linhas))
    if canal_csv is not None:
        sim = sim.merge(canal_csv[["video_id", "views", "views_ate_dia2"]].rename(columns={"video_id": "id"}), on="id", how="left")
    return sim


def relativo(tabela, colunas):
    """Cada feature menos a média do próprio canal (só features, sem rótulo)."""
    rel = tabela.reindex(columns=colunas).astype(float)   # coluna ausente (ex.: vídeo sem legenda) vira NaN e é imputada
    return rel - rel.groupby(tabela["canal"]).transform("mean")


# ---------- modelos ----------
MODELOS = {
    "Regressão Logística": (LogisticRegression(max_iter=3000, class_weight="balanced"), {"m__C": [0.01, 0.1, 1]}),
    "KNN": (KNeighborsClassifier(), {"m__n_neighbors": [15, 31, 51]}),
    "Árvore de Decisão": (DecisionTreeClassifier(class_weight="balanced", random_state=SEMENTE),
                          {"m__max_depth": [2, 3, 5], "m__min_samples_leaf": [10, 20]}),
    "Random Forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEMENTE, n_jobs=1),
                      {"m__max_depth": [3, 5, None], "m__min_samples_leaf": [5, 15]}),
}


def pipe(est):
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler()), ("m", est)])


def cv_aninhada(X, y, g, nome):
    est, grade = MODELOS[nome]
    externo = GroupKFold(n_splits=5)
    prob = np.zeros(len(y))
    aucs = []
    for tr, te in externo.split(X, y, g):
        busca = GridSearchCV(pipe(est), grade, scoring="roc_auc", cv=GroupKFold(n_splits=4), n_jobs=-1)
        busca.fit(X.iloc[tr], y[tr], groups=g[tr])
        prob[te] = busca.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob[te]))
    return np.mean(aucs), np.std(aucs), prob


if __name__ == "__main__":
    base = montar_base()
    y = (base["percentil"] >= 0.8).astype(int).values
    g = base["canal"].values
    print(f"Base: {len(base)} Shorts, {base['canal'].nunique()} canais, viral (topo 20% do canal) = {y.mean():.1%}; "
          f"com legenda: {base['rot_tem_legenda'].mean():.0%}; com capa: {base['capa_brilho'].notna().mean():.0%}")
    todas = [c for cols in GRUPOS.values() for c in cols]
    X = relativo(base, todas)

    print("\n== Ablação (Random Forest e Logística, AUC na validação cruzada por canal) ==")
    conjuntos = {"metadados (como o v1)": ["metadados"], "+ título": ["metadados", "título"],
                 "+ capa": ["metadados", "título", "capa"], "+ roteiro": ["metadados", "título", "capa", "roteiro"],
                 "tudo (+ cenas)": list(GRUPOS)}
    for gname in GRUPOS:
        conjuntos[f"só {gname}"] = [gname]
    if os.environ.get("RAPIDO"):
        conjuntos = dict(list(conjuntos.items())[:1])
        MODELOS.pop("KNN"); MODELOS.pop("Árvore de Decisão")
    ablacao = []
    for rotulo, grupos in conjuntos.items():
        cols = [c for gr in grupos for c in GRUPOS[gr]]
        for nome in ["Random Forest", "Regressão Logística"]:
            m, s, _ = cv_aninhada(X[cols], y, g, nome)
            ablacao.append({"conjunto": rotulo, "modelo": nome, "auc": round(m, 3), "dp": round(s, 3)})
            print(f"{rotulo:24s} {nome:20s} AUC {m:.3f} ± {s:.3f}")

    print("\n== Os 4 modelos com todas as features ==")
    comparacao, probs = [], {}
    for nome in MODELOS:
        m, s, p = cv_aninhada(X, y, g, nome)
        probs[nome] = p
        comparacao.append({"modelo": nome, "auc": round(m, 3), "dp": round(s, 3)})
        print(f"{nome:20s} AUC {m:.3f} ± {s:.3f}")
    melhor = max(comparacao, key=lambda r: r["auc"])["modelo"]
    print("Melhor:", melhor)

    # modelo final em toda a base + importância por grupo (permutação por grupo, na previsão fora da dobra)
    est, grade = MODELOS[melhor]
    final = GridSearchCV(pipe(est), grade, scoring="roc_auc", cv=GroupKFold(n_splits=5), n_jobs=-1).fit(X, y, groups=g)
    print("Hiperparâmetros:", final.best_params_)
    rng = np.random.default_rng(SEMENTE)
    imp_grupo = {}
    for gname, cols in GRUPOS.items():
        quedas = []
        for tr, te in GroupKFold(5).split(X, y, g):
            mod = pipe(est).set_params(**final.best_params_).fit(X.iloc[tr], y[tr])
            base_auc = roc_auc_score(y[te], mod.predict_proba(X.iloc[te])[:, 1])
            for _ in range(3 if os.environ.get("RAPIDO") else 10):
                Xp = X.iloc[te].copy()
                Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
                quedas.append(base_auc - roc_auc_score(y[te], mod.predict_proba(Xp)[:, 1]))
        imp_grupo[gname] = round(float(np.mean(quedas)), 4)
    print("Importância por grupo (queda no AUC):", imp_grupo)
    imp = permutation_importance(final.best_estimator_, X, y, scoring="roc_auc", n_repeats=10, random_state=SEMENTE)
    top = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(12).round(4)
    print("Top features (permutação na base inteira):\n", top.to_string())

    # teste no canal
    sim = montar_simulando()
    Xs = relativo(sim, todas)
    sim["prob_v2"] = final.predict_proba(Xs)[:, 1]
    ok = sim["views"].notna()
    rho, p = spearmanr(sim.loc[ok, "prob_v2"], sim.loc[ok, "views"])
    rho2 = spearmanr(sim["prob_v2"], sim["views_ate_dia2"], nan_policy="omit")
    print(f"\n== Simulando ({len(sim)} Shorts) == Spearman(prob, views) = {rho:.3f} (p = {p:.3f}); "
          f"com views até o 2º dia: {rho2[0]:.3f} (p = {rho2[1]:.3f}, n = {sim['views_ate_dia2'].notna().sum()})")
    print(sim.sort_values("prob_v2", ascending=False)[["titulo", "prob_v2", "views", "cen_ecr", "rot_palavras_min", "capa_brilho"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    json.dump({"base": {"n": len(base), "canais": int(base["canal"].nunique()), "viral": float(y.mean())},
               "ablacao": ablacao, "comparacao": comparacao, "melhor": melhor, "params": {k: str(v) for k, v in final.best_params_.items()},
               "imp_grupo": imp_grupo, "top": top.to_dict(),
               "simulando": {"rho": float(rho), "p": float(p), "rho_dia2": float(rho2[0]), "p_dia2": float(rho2[1]),
                             "videos": sim[["id", "titulo", "prob_v2", "views", "views_ate_dia2"]].to_dict("records")}},
              open(S / "resultado_v2.json", "w"), ensure_ascii=False, indent=1, default=float)
    base.to_csv(S / "base_v2.csv", index=False)
    sim.to_csv(S / "simulando_v2.csv", index=False)
