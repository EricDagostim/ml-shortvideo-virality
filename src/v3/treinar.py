"""Treina e avalia o preditor v3: "este Short vai estourar (>= 5x a régua do canal)?", só com o que existe antes de postar.

Avaliação honesta
- Dobras por CANAL (5): nenhum canal aparece no treino e no teste. É o caso do Simulando, que não está na base.
- Demanda (vizinhos) e cabeças lineares calculadas dentro de cada dobra, só com canais do treino; para as linhas de
  treino, as cabeças saem de uma validação aninhada (também por canal), então nenhuma linha vê o próprio rótulo.
- Limiar de decisão de cada dobra escolhido nas OUTRAS dobras (cross-fitting): a precisão reportada não foi ajustada
  nos mesmos vídeos em que é medida.
- AUC dentro do canal: separa os Shorts de um mesmo canal (é a decisão do criador), não só os canais entre si.
- Teste temporal: treina com o passado e testa nos Shorts publicados depois de uma data, em canais nunca vistos.

Uso: .venv/bin/python -m src.v3.treinar [--rapido]
Saída: reports/v3/validacao.json, importancia_ganho.csv, data/v3/oof.parquet e modelos/v3/modelo.joblib
"""
import argparse
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

from src.v3 import embeddings as emb
from src.v3.config import MODELOS, MULT_VIRAL, RELATORIO, SEMENTE, V3, VIEWS_MIN_VIRAL
from src.v3.features import CABECAS, GRUPOS, KNN_COLS, QUADROS, demanda, lingua_cod, montar_X

INICIO = "2022-01-01"          # antes disso a aba Shorts mistura vídeos verticais antigos (o formato é de 2021)
ALVOS_PRECISAO = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
# Árvores pequenas e bem regularizadas generalizaram melhor para canais nunca vistos que as de 63 folhas
# (data/v3/exp/variantes.log). O modelo é um PAR: classificador (estoura ou não) + regressão do múltiplo da régua.
PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=15, min_child_samples=400, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=30.0, verbose=-1, seed=SEMENTE, num_threads=10)
PARAMS_REG = dict(PARAMS, objective="huber", alpha=1.0)
ALVO_REG = (np.log(0.1), np.log(50))        # o múltiplo da régua (log) é cortado em 0,1x e 50x para a regressão
ALPHA_RIDGE = 30.0


# ----------------------------------------------------------------------------------------------- dados
def carregar_base():
    df = pd.read_parquet(V3 / "base_v3.parquet")
    ids = df["id"].tolist()
    Et, tem_t, _ = emb.carregar("titulos", ids)
    Ec, tem_c, anim = emb.carregar("capas", ids)
    Q, tem_q = emb.carregar_quadros(ids)
    df["capa_animado"] = anim if anim is not None else np.nan
    qf = V3 / "quadros_feats.jsonl"
    if qf.exists():
        q = pd.DataFrame([json.loads(l) for l in open(qf)])
        q = q[q.get("erro").isna()] if "erro" in q else q
        df = df.merge(q.drop(columns=[c for c in ("erro", "qd_n", "qd_capa_sim") if c in q]).drop_duplicates("id"),
                      on="id", how="left")
    for c in QUADROS:
        if c not in df:
            df[c] = np.nan
    df["lingua_cod"] = lingua_cod(df["lingua"])
    df = df[tem_t].reset_index(drop=True)
    return df, Et[tem_t], Ec[tem_t], tem_c[tem_t], Q[tem_t], tem_q[tem_t]


LIMIAR_ANIMADO = 0.6   # canal "animado": em média, os quadros (ou as capas) parecem animação/3D/desenho para o CLIP


def preparar(so_animados=False):
    df, Et, Ec, tem_c, Q, tem_q = carregar_base()
    X, R, proj = montar_X(df, Et, Ec, tem_c, Q, tem_q)
    anim = df["qd_animado"].fillna(df["capa_animado"])
    df["canal_animado"] = anim.groupby(df["canal_id"]).transform("mean")
    vet = {"titulo": np.hstack([Et, R]), "capa": np.nan_to_num(Ec), "quadros": np.nan_to_num(Q)}
    msk = {"titulo": np.ones(len(df), bool), "capa": tem_c, "quadros": tem_q}
    treino = (df["no_treino"] & (df["publicado"] >= pd.Timestamp(INICIO, tz="UTC"))).to_numpy()
    if so_animados:
        treino = treino & (df["canal_animado"] >= LIMIAR_ANIMADO).to_numpy()
    return df, X, Et, vet, msk, proj, treino


def colunas(grupos):
    return [c for g in grupos for c in GRUPOS[g]]


def dobras_por_canal(canais, k=5, semente=SEMENTE):
    u = np.unique(canais)
    rng = np.random.default_rng(semente)
    rng.shuffle(u)
    mapa = {c: i % k for i, c in enumerate(u)}
    f = np.array([mapa[c] for c in canais])
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def pesos(canais):
    """Cada canal pesa ~ raiz do nº de Shorts: canal com 600 Shorts não domina o treino."""
    n = pd.Series(canais).map(pd.Series(canais).value_counts()).to_numpy(float)
    w = 1 / np.sqrt(n)
    return w / w.mean()


def tempo_ns(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s, utc=True).dt.tz_convert(None).astype("datetime64[ns]").astype("int64").to_numpy()


def alvo_cabeca(lo):
    return np.clip(lo, -3, 4)


# ----------------------------------------------------------------------------------------------- cabeças
def ajustar_cabecas(vet, msk, idx, lo, w):
    out = {}
    for nome in ("titulo", "capa", "quadros"):
        m = idx[msk[nome][idx]]
        out[nome] = Ridge(alpha=ALPHA_RIDGE).fit(vet[nome][m], alvo_cabeca(lo[m]), sample_weight=w[m]) if len(m) > 500 else None
    return out


def prever_cabecas(cab, vet, msk, idx):
    out = np.full((len(idx), 3), np.nan, np.float32)
    for j, nome in enumerate(("titulo", "capa", "quadros")):
        if cab.get(nome) is None:
            continue
        m = msk[nome][idx]
        if m.any():
            out[m, j] = cab[nome].predict(vet[nome][idx[m]])
    return out


def cabecas_aninhadas(vet, msk, tr, te, lo, w, canais):
    """Cabeças para o teste (ajustadas no treino) e para o próprio treino (validação aninhada por canal)."""
    cab = ajustar_cabecas(vet, msk, tr, lo, w)
    h_te = prever_cabecas(cab, vet, msk, te)
    h_tr = np.full((len(tr), 3), np.nan, np.float32)
    for a, b in dobras_por_canal(canais[tr], k=5, semente=SEMENTE + 3):
        c = ajustar_cabecas(vet, msk, tr[a], lo, w)
        h_tr[b] = prever_cabecas(c, vet, msk, tr[b])
    return h_tr, h_te


# ----------------------------------------------------------------------------------------------- modelo
def ajustar_lgb(X, y, w, canais, n_iter=None, reg=False):
    """Com n_iter=None, separa 15% dos canais do treino para parar cedo (classificador: average precision;
    regressão: perda huber)."""
    p, Modelo = (PARAMS_REG, lgb.LGBMRegressor) if reg else (PARAMS, lgb.LGBMClassifier)
    if n_iter is None:
        u = np.unique(canais)
        rng = np.random.default_rng(SEMENTE + 1)
        val = set(rng.choice(u, max(1, int(0.15 * len(u))), replace=False))
        mv = np.array([c in val for c in canais])
        m = Modelo(n_estimators=3000, **p)
        m.fit(X[~mv], y[~mv], sample_weight=w[~mv], eval_set=[(X[mv], y[mv])], eval_sample_weight=[w[mv]],
              **({} if reg else {"eval_metric": "average_precision"}), callbacks=[lgb.early_stopping(150, verbose=False)])
        n_iter = max(100, m.best_iteration_ or 300)
    m = Modelo(n_estimators=n_iter, **p)
    m.fit(X, y, sample_weight=w)
    return m, n_iter


def ajustar_par(X, y, lo, w, canais, n_iter=(None, None)):
    mb, nb = ajustar_lgb(X, y, w, canais, n_iter[0])
    mr, nr = ajustar_lgb(X, np.clip(lo, *ALVO_REG), w, canais, n_iter[1], reg=True)
    return (mb, mr), (nb, nr)


def prever_par(mods, X):
    return mods[0].predict_proba(X)[:, 1], mods[1].predict(X)


def combinar(pb, pr, ref_b, ref_r):
    """Nota final: média dos postos (0-1) da chance do classificador e do múltiplo previsto pela regressão, medidos na
    distribuição fora da amostra (ref_* ordenados). O par acerta mais no topo do que cada um sozinho."""
    return 0.5 * (np.searchsorted(ref_b, pb, side="right") / len(ref_b)
                  + np.searchsorted(ref_r, pr, side="right") / len(ref_r))


# ----------------------------------------------------------------------------------------------- métricas
def precisao_no_topo(y, p, fracs=(0.005, 0.01, 0.02, 0.05, 0.10)):
    ordem = np.argsort(-p)
    out = {}
    for f in fracs:
        k = max(1, int(round(f * len(y))))
        top = ordem[:k]
        out[f"prec_top{f * 100:g}%"] = float(y[top].mean())
        out[f"recall_top{f * 100:g}%"] = float(y[top].sum() / max(1, y.sum()))
    return out


def limiar_para_precisao(y, p, alvo, min_pos=20):
    """Menor limiar cuja precisão (entre os acima dele) é >= alvo, com pelo menos min_pos previsões positivas."""
    ordem = np.argsort(-p)
    ys, ps = y[ordem], p[ordem]
    prec = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    ok = np.where((prec >= alvo) & (np.arange(1, len(ys) + 1) >= min_pos))[0]
    if not len(ok):
        return None
    return float(ps[ok.max()])


def limiar_nao_viral(y, p, fator=0.4):
    """Maior limiar abaixo do qual a taxa de viral fica <= fator x taxa base."""
    ordem = np.argsort(p)
    taxa = np.cumsum(y[ordem]) / np.arange(1, len(ordem) + 1)
    ok = np.where((taxa <= y.mean() * fator) & (np.arange(1, len(ordem) + 1) >= 200))[0]
    return float(p[ordem][ok.max()]) if len(ok) else None


def cross_fit(y, p, dobra, alvo=None, nao=False):
    """Limiar escolhido nas outras dobras, aplicado na dobra: precisão e recall sem otimismo."""
    tp = fp = n_sel = 0
    for k in np.unique(dobra):
        fora, dentro = dobra != k, dobra == k
        t = limiar_nao_viral(y[fora], p[fora]) if nao else limiar_para_precisao(y[fora], p[fora], alvo)
        if t is None:
            continue
        sel = (p[dentro] <= t) if nao else (p[dentro] >= t)
        n_sel += int(sel.sum())
        tp += int((sel & (y[dentro] == 1)).sum())
        fp += int((sel & (y[dentro] == 0)).sum())
    if nao:
        return {"previstos_nao_virais": n_sel, "virais_entre_eles": tp, "taxa_viral_entre_eles": tp / max(1, n_sel),
                "cobertura": n_sel / len(y), "virais_perdidos_frac": tp / max(1, int(y.sum()))}
    return {"alvo": alvo, "previstos_virais": tp + fp, "acertos": tp, "falsos_positivos": fp,
            "precisao": tp / (tp + fp) if tp + fp else None, "recall": tp / max(1, int(y.sum())),
            "cobertura": (tp + fp) / len(y)}


def auc_dentro(canais, y, p):
    aucs, ns = [], []
    d = pd.DataFrame({"c": canais, "y": y, "p": p})
    for _, g in d.groupby("c"):
        if g["y"].sum() >= 3 and (1 - g["y"]).sum() >= 3:
            aucs.append(roc_auc_score(g["y"], g["p"]))
            ns.append(len(g))
    return {"auc_dentro_canal": float(np.mean(aucs)), "auc_dentro_canal_pond": float(np.average(aucs, weights=ns)),
            "canais": len(aucs)}


def resumo(y, p, canais=None, dobra=None):
    r = {"n": int(len(y)), "taxa_viral": float(y.mean()), "auc": float(roc_auc_score(y, p)),
         "ap": float(average_precision_score(y, p)), **precisao_no_topo(y, p)}
    if canais is not None:
        r.update(auc_dentro(canais, y, p))
    if dobra is not None:
        r["decisao_viral"] = [cross_fit(y, p, dobra, a) for a in ALVOS_PRECISAO]
        r["decisao_nao_viral"] = cross_fit(y, p, dobra, nao=True)
    return r


# ----------------------------------------------------------------------------------------------- validação
class Dobras:
    """Guarda, por dobra, demanda e cabeças (o caro) para reaproveitar entre ablações."""

    def __init__(self, df, X, Et, vet, msk, treino):
        self.D = df[treino].reset_index(drop=True)
        self.X = X[treino].reset_index(drop=True)
        self.E = Et[treino]
        self.vet = {k: v[treino] for k, v in vet.items()}
        self.msk = {k: v[treino] for k, v in msk.items()}
        self.y = self.D["viral"].to_numpy(int)
        self.lo = self.D["log_outlier"].to_numpy(float)
        self.canais = self.D["canal_id"].to_numpy()
        self.cod = pd.factorize(self.canais)[0]
        self.t = tempo_ns(self.D["publicado"])
        self.w = pesos(self.canais)

    def extras(self, tr, te):
        y, lo, t, cod, E = self.y, self.lo, self.t, self.cod, self.E
        d_tr = demanda(E[tr], t[tr], cod[tr], E[tr], t[tr], cod[tr], lo[tr], y[tr].astype(float))
        d_te = demanda(E[te], t[te], cod[te], E[tr], t[tr], cod[tr], lo[tr], y[tr].astype(float))
        h_tr, h_te = cabecas_aninhadas(self.vet, self.msk, tr, te, lo, self.w, self.canais)
        return np.hstack([d_tr, h_tr]), np.hstack([d_te, h_te])

    def matriz(self, idx, extra):
        return pd.concat([self.X.iloc[idx].reset_index(drop=True), pd.DataFrame(extra, columns=KNN_COLS + CABECAS)], axis=1)


def validar(dd: Dobras, grupos_lista):
    dobras = dobras_por_canal(dd.canais)
    dobra = np.zeros(len(dd.y), int)
    ob = {nome: np.full(len(dd.y), np.nan) for nome in grupos_lista}
    orr = {nome: np.full(len(dd.y), np.nan) for nome in grupos_lista}
    iters = []
    for k, (tr, te) in enumerate(dobras):
        t0 = time.time()
        dobra[te] = k
        e_tr, e_te = dd.extras(tr, te)
        A, B = dd.matriz(tr, e_tr), dd.matriz(te, e_te)
        for nome, grupos in grupos_lista.items():
            cols = colunas(grupos)
            mods, n = ajustar_par(A[cols].to_numpy(), dd.y[tr], dd.lo[tr], dd.w[tr], dd.canais[tr])
            ob[nome][te], orr[nome][te] = prever_par(mods, B[cols].to_numpy())
            if nome == "tudo":
                iters.append(n)
        print(f"dobra {k + 1}/5: {len(te)} Shorts de teste, {time.time() - t0:.0f} s; "
              f"AUC tudo (classificador) {roc_auc_score(dd.y[te], ob['tudo'][te]):.3f}", flush=True)
    oof = {nome: combinar(ob[nome], orr[nome], np.sort(ob[nome]), np.sort(orr[nome])) for nome in grupos_lista}
    n_iter = (int(np.median([i[0] for i in iters])), int(np.median([i[1] for i in iters])))
    return dobra, oof, n_iter, (ob["tudo"], orr["tudo"])


def teste_temporal(dd: Dobras, corte="2026-06-01"):
    """Treina com Shorts de 70% dos canais publicados até 14 dias antes do corte; testa nos Shorts dos outros 30%
    publicados depois do corte."""
    c = pd.Timestamp(corte, tz="UTC").value
    u = np.unique(dd.canais)
    rng = np.random.default_rng(7)
    teste_c = set(rng.choice(u, int(0.3 * len(u)), replace=False))
    em_teste = np.array([x in teste_c for x in dd.canais])
    tr = np.where(~em_teste & (dd.t < c - 14 * 86400e9))[0]
    te = np.where(em_teste & (dd.t >= c))[0]
    e_tr, e_te = dd.extras(tr, te)
    cols = colunas(list(GRUPOS))
    A, B = dd.matriz(tr, e_tr)[cols], dd.matriz(te, e_te)[cols]
    mods, _ = ajustar_par(A.to_numpy(), dd.y[tr], dd.lo[tr], dd.w[tr], dd.canais[tr])
    pb, pr = prever_par(mods, B.to_numpy())
    p = combinar(pb, pr, np.sort(pb), np.sort(pr))
    r = resumo(dd.y[te], p, dd.canais[te])
    r.update({"corte": corte, "n_treino": int(len(tr)), "canais_teste": int(len(set(dd.canais[te])))})
    return r


# ----------------------------------------------------------------------------------------------- final
def escolher_alvo_viral(decisoes, min_previstos=100):
    """Alvo de precisão do veredito VIRAL: o de maior precisão FORA da amostra entre os que marcam >= min_previstos."""
    ok = [d for d in decisoes if d["precisao"] is not None and d["previstos_virais"] >= min_previstos]
    return str(max(ok, key=lambda d: (round(d["precisao"], 2), -d["alvo"]))["alvo"]) if ok else "0.3"


def treinar_final(dd: Dobras, proj, n_iter, oof, dobra, oof_par, alvo_viral="0.3"):
    todos = np.arange(len(dd.y))
    d = demanda(dd.E, dd.t, dd.cod, dd.E, dd.t, dd.cod, dd.lo, dd.y.astype(float))
    cab = ajustar_cabecas(dd.vet, dd.msk, todos, dd.lo, dd.w)
    h = np.full((len(todos), 3), np.nan, np.float32)            # no treino final, as cabeças das linhas de treino
    for a, b in dobras_por_canal(dd.canais, k=5, semente=SEMENTE + 3):   # vêm de validação aninhada (como no CV)
        h[b] = prever_cabecas(ajustar_cabecas(dd.vet, dd.msk, a, dd.lo, dd.w), dd.vet, dd.msk, b)
    cols = colunas(list(GRUPOS))
    Xf = dd.matriz(todos, np.hstack([d, h]))[cols]
    mods, _ = ajustar_par(Xf.to_numpy(), dd.y, dd.lo, dd.w, dd.canais, n_iter=n_iter)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof, dd.y)
    limiares = {str(a): limiar_para_precisao(dd.y, oof, a) for a in ALVOS_PRECISAO}
    MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": mods[0], "regressao": mods[1], "ref_b": np.sort(oof_par[0]).astype(np.float32),
                 "ref_r": np.sort(oof_par[1]).astype(np.float32), "colunas": cols, "proj": proj, "cabecas": cab,
                 "isotonica": iso,
                 "limiares_precisao": limiares, "limiar_nao_viral": limiar_nao_viral(dd.y, oof),
                 "taxa_base": float(dd.y.mean()), "n_iter": n_iter, "mult_viral": MULT_VIRAL, "inicio": INICIO,
                 "views_min_viral": VIEWS_MIN_VIRAL, "alvo_viral": alvo_viral, "alvo_promissor": "0.2",
                 "oof": {"p": oof.astype(np.float32), "y": dd.y.astype(np.int8)}}, MODELOS / "modelo.joblib")
    return pd.Series(mods[0].booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)


def _fmt(r):
    return (f"AUC {r['auc']:.3f}  AP {r['ap']:.3f} (base {r['taxa_viral']:.3f})  dentro do canal {r.get('auc_dentro_canal', float('nan')):.3f}  "
            f"prec top0.5% {r['prec_top0.5%']:.2f} top1% {r['prec_top1%']:.2f} top5% {r['prec_top5%']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--so-animados", action="store_true", help="treina e avalia só com canais animados")
    ap.add_argument("--amostra", type=int, help="teste rápido: só N canais (não use para o modelo de verdade)")
    a = ap.parse_args()
    t0 = time.time()
    df, X, Et, vet, msk, proj, treino = preparar(a.so_animados)
    if a.amostra:
        u = df.loc[treino, "canal_id"].unique()
        keep = set(np.random.default_rng(0).choice(u, min(a.amostra, len(u)), replace=False))
        treino = treino & df["canal_id"].isin(keep).to_numpy()
    dd = Dobras(df, X, Et, vet, msk, treino)
    print(f"treino: {len(dd.y)} Shorts de {len(set(dd.canais))} canais, viral {dd.y.mean():.2%}; com capa "
          f"{dd.msk['capa'].mean():.0%}, com quadros {dd.msk['quadros'].mean():.0%}  [{time.time() - t0:.0f} s]", flush=True)
    todos = list(GRUPOS)
    grupos = {"canal": ["canal"], "canal + meta + título": ["canal", "meta", "título"],
              "+ capa": ["canal", "meta", "título", "capa"], "+ quadros": ["canal", "meta", "título", "capa", "quadros"],
              "+ novidade": ["canal", "meta", "título", "capa", "quadros", "novidade"], "tudo": todos,
              "só conteúdo (sem canal)": [g for g in todos if g != "canal"]}
    if a.rapido:
        grupos = {"canal": ["canal"], "tudo": todos}
    dobra, oof, n_iter, oof_par = validar(dd, grupos)
    RELATORIO.mkdir(parents=True, exist_ok=True)
    res = {"treino": {"n": int(len(dd.y)), "canais": int(len(set(dd.canais))), "inicio": INICIO, "mult_viral": MULT_VIRAL}}
    sub = {"canais_animados": (dd.D["canal_animado"] >= LIMIAR_ANIMADO).to_numpy(), "portugues": (dd.D["lingua"] == "pt").to_numpy(),
           "canal_pequeno_regua_lt_5k": (dd.D["regua"] < 5000).to_numpy(),
           "animado_pequeno": ((dd.D["canal_animado"] >= LIMIAR_ANIMADO) & (dd.D["regua"] < 5000)).to_numpy()}
    for nome, p in oof.items():
        res[nome] = resumo(dd.y, p, dd.canais, dobra if nome == "tudo" else None)
        for s, m in sub.items():
            res[nome][s] = resumo(dd.y[m], p[m], dd.canais[m], dobra[m] if nome == "tudo" else None) if m.sum() > 300 else None
        print(f"{nome:26s} {_fmt(res[nome])}", flush=True)
    for s in sub:
        if res["tudo"][s]:
            print(f"  tudo | {s:24s} n={res['tudo'][s]['n']:6d} {_fmt(res['tudo'][s])}")
    print("\nDecisão VIRAL no 'tudo' (limiar escolhido nas outras dobras):")
    for d in res["tudo"]["decisao_viral"]:
        print(f"  precisão alvo {d['alvo']:.0%}: {d['previstos_virais']} previstos ({d['cobertura']:.2%} dos Shorts), {d['acertos']} acertos, "
              f"{d['falsos_positivos']} falsos positivos -> precisão {d['precisao'] if d['precisao'] is None else round(d['precisao'], 3)}, "
              f"recall {d['recall']:.3f}")
    print("Decisão NÃO VIRAL:", {k: round(v, 3) for k, v in res["tudo"]["decisao_nao_viral"].items()})
    if not a.rapido:
        res["teste_temporal"] = teste_temporal(dd)
        print("\nTeste temporal:", _fmt(res["teste_temporal"]), "| canais de teste", res["teste_temporal"]["canais_teste"])
    alvo_viral = escolher_alvo_viral(res["tudo"]["decisao_viral"])
    res["alvo_viral"] = alvo_viral
    print("Alvo do veredito VIRAL:", alvo_viral)
    imp = treinar_final(dd, proj, n_iter, oof["tudo"], dobra, oof_par, alvo_viral)
    imp.to_csv(RELATORIO / "importancia_ganho.csv")
    g = {gr: float(imp[[c for c in cols if c in imp.index]].sum() / imp.sum()) for gr, cols in GRUPOS.items()}
    res["importancia_grupo"] = g
    print("\nImportância por grupo (ganho):", {k: round(v, 3) for k, v in g.items()})
    print("Top 15 features:\n", (imp / imp.sum()).head(15).round(4).to_string())
    json.dump(res, open(RELATORIO / "validacao.json", "w"), ensure_ascii=False, indent=1)
    pd.DataFrame({"id": dd.D["id"], "canal_id": dd.canais, "dobra": dobra, "viral": dd.y, "log_outlier": dd.lo,
                  **{f"p_{k}": v for k, v in oof.items()}}).to_parquet(V3 / "oof.parquet", index=False)
    print(f"\nfim em {(time.time() - t0) / 60:.1f} min")
