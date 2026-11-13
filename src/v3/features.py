"""Matriz de features do v3 (tudo o que se sabe antes de postar).

Grupos
  canal     contexto do próprio canal até a véspera (base.py)
  título    padrões do texto (hipótese, pergunta, número, extremo...) e componentes do vetor multilíngue
  meta      duração, hora, dia, descrição, tags, língua
  capa      componentes do CLIP da capa e nota "animado"
  quadros   o conteúdo do vídeo: 3 quadros a 25/50/75% da duração (CLIP, animado, diversidade visual, cor, texto na tela)
  novidade  quanto o título/capa se parece com o que o canal já postou, e como foram os parecidos do próprio canal
  demanda   vizinhos: os títulos mais parecidos de OUTROS canais, publicados ANTES — quantos estouraram e quanto
            (é a pergunta "esse tema já provou que estoura?"). Calculado dentro de cada dobra, só com canais do treino.
  cabeças   uma regressão linear (ridge) sobre o vetor inteiro de título, capa e quadros, prevendo o múltiplo da régua;
            a previsão dela entra como feature (empilhamento com validação aninhada, sem vazamento)
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.v3.config import MATURO_DIAS, MULT_VIRAL

# Fora de propósito: ctx_n_anteriores e ctx_idade_canal_dias. O treino só tem canais com >= 30 Shorts anteriores e
# >= 90 dias (viés de sobrevivência, ver config.py); num canal novo essas duas só extrapolariam.
CANAL = ["ctx_log_regua", "ctx_dispersao", "ctx_p90_rel", "ctx_taxa_estouro", "ctx_taxa_2x", "ctx_max_outlier",
         "ctx_ultimo_outlier", "ctx_tendencia", "ctx_dias_desde_ultimo", "ctx_shorts_30d", "ctx_dur_rel"]
TITULO = ["tit_caracteres", "tit_palavras", "tit_interrogacao", "tit_exclamacao", "tit_reticencias", "tit_numeros", "tit_emojis",
          "tit_maiusculas", "tit_hashtags", "tit_tem_shorts", "tit_hipotese", "tit_como_porque", "tit_voce", "tit_extremo",
          "tit_tempo", "tit_perigo"]
META = ["duracao_s", "pub_hora_utc", "pub_dia_semana", "desc_caracteres", "desc_hashtags", "desc_links", "n_tags", "lingua_cod"]
LINGUAS = ["en", "pt", "es", "de", "fr", "hi", "it", "ru", "ar", "id", "ja", "ko", "tr"]
N_PCA_TIT, N_PCA_CAPA = 24, 16


def lingua_cod(s: pd.Series) -> pd.Series:
    return s.map({l: i for i, l in enumerate(LINGUAS)}).fillna(len(LINGUAS)).astype(int)


def novidade(df: pd.DataFrame, Et: np.ndarray, Ec: np.ndarray, tem_c: np.ndarray) -> pd.DataFrame:
    """Parecença com os Shorts anteriores do próprio canal e o resultado dos mais parecidos (rótulos do passado)."""
    out = np.full((len(df), 7), np.nan, np.float32)
    lo = df["log_outlier"].to_numpy(float)
    maduro = df["idade_dias"].to_numpy(float) >= 7
    for _, idx in df.groupby("canal_id", sort=False).indices.items():
        idx = idx[np.argsort(df["publicado"].to_numpy()[idx], kind="stable")]
        t = df["publicado"].to_numpy()[idx]
        E, C = Et[idx], Ec[idx]
        S = E @ E.T
        Sc = np.nan_to_num(C) @ np.nan_to_num(C).T
        for j in range(1, len(idx)):
            a0 = max(0, j - 30)
            s = S[j, a0:j]
            out[idx[j], 0] = s.max()
            out[idx[j], 1] = s.mean()
            if tem_c[idx[j]] and tem_c[idx[a0:j]].any():
                sc = Sc[j, a0:j][tem_c[idx[a0:j]]]
                out[idx[j], 2] = sc.mean()
                out[idx[j], 3] = sc.max()
            # os 3 Shorts anteriores do canal mais parecidos (todo o histórico) cujo resultado já se sabia: publicados
            # pelo menos MATURO_DIAS antes deste
            ant = np.arange(j)
            ok = ~np.isnan(lo[idx[ant]]) & maduro[idx[ant]] & ((t[j] - t[ant]) >= np.timedelta64(MATURO_DIAS, "D"))
            if ok.sum() >= 3:
                ss = S[j, ant[ok]]
                top = np.argsort(-ss)[:3]
                out[idx[j], 4] = lo[idx[ant[ok][top]]].mean()
                out[idx[j], 5] = ss[top].mean()
                out[idx[j], 6] = (lo[idx[ant[ok][top]]] >= np.log(MULT_VIRAL)).mean()
    return pd.DataFrame(out, index=df.index, columns=["nov_tit_max", "nov_tit_media", "nov_capa_media", "nov_capa_max",
                                                       "nov_parecidos_log_outlier", "nov_parecidos_sim", "nov_parecidos_viral"])


KNN_COLS = ["dem_log_outlier", "dem_taxa_viral", "dem_sim_max", "dem_sim_media", "dem_n_proximos", "dem_viral_proximos",
            "dem_rec_log_outlier", "dem_rec_taxa_viral", "dem_rec_sim_max", "dem_rec_n_proximos"]


GAP_DEMANDA_DIAS = 14   # o vizinho precisa ter saído 14 dias antes: é a idade mínima de um Short da base na hora de usar


def _knn_passada(Eq, tq, cq, Ep, tp, cp, yp, vp, k, janela_dias=None, bloco=768):
    nq = len(Eq)
    out = np.full((nq, 6), np.nan, np.float32)
    jan = np.int64(janela_dias * 86400e9) if janela_dias else None
    gap = np.int64(GAP_DEMANDA_DIAS * 86400e9)
    for s in range(0, nq, bloco):
        e = min(nq, s + bloco)
        S = Eq[s:e] @ Ep.T
        proibido = (tp[None, :] > tq[s:e, None] - gap) | (cp[None, :] == cq[s:e, None])
        if jan is not None:
            proibido |= (tq[s:e, None] - tp[None, :]) > jan
        S[proibido] = -9
        kk = min(k, S.shape[1] - 1)
        idx = np.argpartition(-S, kk, axis=1)[:, :kk]
        sims = np.take_along_axis(S, idx, 1)
        ok = sims > -2
        w = np.where(ok, np.clip(sims, 0.01, None) ** 3, 0)
        ws = w.sum(1)
        y = yp[idx]
        v = vp[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:e, 0] = (w * y).sum(1) / ws
            out[s:e, 1] = (w * v).sum(1) / ws
            out[s:e, 2] = np.where(ok, sims, -9).max(1)
            out[s:e, 3] = np.where(ok, sims, 0).sum(1) / ok.sum(1)
            prox = ok & (sims >= 0.7)
            out[s:e, 4] = prox.sum(1)
            out[s:e, 5] = np.where(prox, v, 0).sum(1) / prox.sum(1)
        out[s:e][ok.sum(1) == 0] = np.nan
    return out


def demanda(Eq, tq, cq, Ep, tp, cp, yp, vp, k=30) -> np.ndarray:
    """Vizinhos de OUTROS canais, publicados ANTES. Duas passadas: todo o passado e só os últimos 120 dias (tendência)."""
    a = _knn_passada(Eq, tq, cq, Ep, tp, cp, yp, vp, k)
    b = _knn_passada(Eq, tq, cq, Ep, tp, cp, yp, vp, k, janela_dias=120)
    return np.concatenate([a, b[:, [0, 1, 2, 4]]], axis=1)


N_PCA_REL, N_PCA_QD = 16, 16
QUADROS = ["qd_animado", "qd_brilho", "qd_saturacao", "qd_colorido", "qd_bordas", "qd_bordas_topo", "qd_bordas_meio",
           "qd_brilho_var", "qd_diversidade", "qd_capa_sim"]


def relativo_canal(df: pd.DataFrame, Et: np.ndarray, janela: int = 30) -> np.ndarray:
    """Título menos a média dos `janela` títulos anteriores do canal: para onde este Short foge do que o canal faz."""
    R = np.zeros_like(Et)
    t = df["publicado"].to_numpy()
    for _, idx in df.groupby("canal_id", sort=False).indices.items():
        idx = idx[np.argsort(t[idx], kind="stable")]
        cs = np.cumsum(Et[idx], 0)
        for j in range(1, len(idx)):
            a0 = max(0, j - janela)
            mu = (cs[j - 1] - (cs[a0 - 1] if a0 > 0 else 0)) / (j - a0)
            R[idx[j]] = Et[idx[j]] - mu
    return R


class Projecoes:
    """PCA dos vetores (não supervisionado: ajustado uma vez com toda a base): título, título relativo ao canal,
    capa e quadros do vídeo."""

    def __init__(self):
        self.p = {"emb_tit": PCA(N_PCA_TIT, random_state=0), "rel_tit": PCA(N_PCA_REL, random_state=0),
                  "emb_capa": PCA(N_PCA_CAPA, random_state=0), "emb_qd": PCA(N_PCA_QD, random_state=0)}
        self.ok = {}

    def fit(self, vetores: dict, mascaras: dict):
        for k, pca in self.p.items():
            m = mascaras[k]
            self.ok[k] = int(m.sum()) > 20 * pca.n_components
            if self.ok[k]:
                pca.fit(vetores[k][m])
        return self

    def transform(self, vetores: dict, mascaras: dict) -> pd.DataFrame:
        partes = []
        for k, pca in self.p.items():
            n = len(vetores[k])
            M = np.full((n, pca.n_components), np.nan, np.float32)
            m = mascaras[k]
            if self.ok.get(k) and m.any():
                M[m] = pca.transform(vetores[k][m])
            partes.append(pd.DataFrame(M, columns=[f"{k}_{i}" for i in range(pca.n_components)]))
        return pd.concat(partes, axis=1)


def montar_X(df: pd.DataFrame, Et, Ec, tem_c, Q, tem_q, proj: Projecoes | None = None):
    """Matriz sem demanda e sem as 'cabeças' lineares (essas dependem dos rótulos e são feitas dentro de cada dobra).
    `df` precisa das colunas de base.py (ctx_*, tit_*, desc_*...) e de capa_animado/qd_*. Devolve (X, R, proj)."""
    R = relativo_canal(df, Et)
    nov = novidade(df, Et, Ec, tem_c)
    vet = {"emb_tit": Et, "rel_tit": R, "emb_capa": np.nan_to_num(Ec), "emb_qd": np.nan_to_num(Q)}
    msk = {"emb_tit": np.ones(len(df), bool), "rel_tit": np.ones(len(df), bool), "emb_capa": tem_c, "emb_qd": tem_q}
    if proj is None:
        proj = Projecoes().fit(vet, msk)
    P = proj.transform(vet, msk)
    base = df.reindex(columns=CANAL + TITULO + META + ["capa_animado"] + QUADROS).reset_index(drop=True)
    sim = np.where(tem_c & tem_q, (np.nan_to_num(Q) * np.nan_to_num(Ec)).sum(1), np.nan)
    base["qd_capa_sim"] = sim
    X = pd.concat([base, nov.reset_index(drop=True), P], axis=1).astype(np.float32)
    return X, R, proj


CABECAS = ["head_titulo", "head_capa", "head_quadros"]

GRUPOS = {
    "canal": CANAL,
    "título": TITULO + [f"emb_tit_{i}" for i in range(N_PCA_TIT)] + [f"rel_tit_{i}" for i in range(N_PCA_REL)] + ["head_titulo"],
    "meta": META,
    "capa": ["capa_animado"] + [f"emb_capa_{i}" for i in range(N_PCA_CAPA)] + ["head_capa"],
    "quadros": QUADROS + [f"emb_qd_{i}" for i in range(N_PCA_QD)] + ["head_quadros"],
    "novidade": ["nov_tit_max", "nov_tit_media", "nov_capa_media", "nov_capa_max", "nov_parecidos_log_outlier",
                 "nov_parecidos_sim", "nov_parecidos_viral"],
    "demanda": KNN_COLS,
}
