import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    COLUNA_ALVO,
    COLUNAS_CRIADOR,
    COLUNAS_TEXTO_CONTAGEM,
    COLUNAS_TEXTO_OUTRAS,
    COLUNAS_VISUAIS,
    FIGURAS_DIR,
    TABELAS_DIR,
)
from src.divisao import carregar_divisao
from src.features import carregar_bruto

sns.set_theme(style="whitegrid", context="notebook")


def _vazio(serie):
    if serie.dtype == object:
        return serie.isna() | (serie.astype(str).str.strip() == "")
    return serie.isna()


def comparar_completude():
    linhas = []
    for plataforma in ("tiktok", "youtube"):
        bruto = carregar_bruto(plataforma)
        for coluna in bruto.columns:
            if coluna.startswith("keyframe") or coluna in ("folder_id", "day"):
                continue
            serie = bruto[coluna]
            vazio = _vazio(serie)
            constante = serie.nunique(dropna=True) <= 1
            dominante = serie.value_counts(normalize=True, dropna=False).iloc[0]
            linhas.append(
                {
                    "plataforma": plataforma,
                    "coluna": coluna,
                    "linhas": len(serie),
                    "pct_vazio": round(100 * vazio.mean(), 2),
                    "pct_valor_mais_frequente": round(100 * dominante, 2),
                    "constante": constante,
                }
            )
    tabela = pd.DataFrame(linhas)
    tiktok = carregar_bruto("tiktok")
    zerado = (
        (tiktok["author_followerCount"] == 0)
        & (tiktok["author_heartCount"] == 0)
        & (tiktok["author_videoCount"] == 0)
    )
    tabela.loc[
        (tabela.plataforma == "tiktok") & tabela.coluna.str.startswith("author_"), "pct_vazio"
    ] = round(100 * zerado.mean(), 2)
    tabela.to_csv(TABELAS_DIR / "completude_plataformas.csv", index=False)
    return tabela


def taxa_viral_por_faixa(treino, coluna, faixas, rotulos):
    faixa = pd.cut(treino[coluna], bins=faixas, labels=rotulos, include_lowest=True, right=True)
    agrupado = treino.groupby(faixa, observed=True)[COLUNA_ALVO].agg(["mean", "size"])
    erro = 1.96 * np.sqrt(agrupado["mean"] * (1 - agrupado["mean"]) / agrupado["size"])
    return agrupado.assign(ic95=erro).rename(columns={"mean": "taxa_viral", "size": "n"})


def grafico_distribuicao(treino, limiar):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(np.log10(treino["visualizacoes"].clip(lower=1)), bins=60, ax=ax, color="#4C72B0")
    ax.axvline(np.log10(limiar), color="#C44E52", ls="--", lw=2, label=f"limiar viral (P80) = {limiar/1e6:.1f} mi")
    ax.set_xlabel("log10(visualizações)")
    ax.set_ylabel("vídeos")
    ax.set_title("TikTok (treino): distribuição de visualizações")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "01_distribuicao_visualizacoes.png", dpi=130)
    plt.close(fig)


def grafico_taxas(treino):
    configuracoes = [
        ("seguidores", [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e9], ["<1 mil", "1-10 mil", "10-100 mil", "100 mil-1 mi", "1-10 mi", ">10 mi"], "Seguidores do criador"),
        ("n_hashtags", [-0.5, 0.5, 2, 4, 7, 12, 400], ["0", "1-2", "3-4", "5-7", "8-12", "13+"], "Nº de hashtags"),
        ("n_hashtags_genericas", [-0.5, 0.5, 1, 2, 100], ["0", "1", "2", "3+"], "Hashtags genéricas (#fyp, #viral...)"),
        ("comprimento_legenda", [-0.5, 0.5, 30, 60, 120, 250, 5000], ["vazia", "1-30", "31-60", "61-120", "121-250", "251+"], "Tamanho da legenda (caracteres)"),
        ("n_emojis", [-0.5, 0.5, 1, 3, 2000], ["0", "1", "2-3", "4+"], "Nº de emojis"),
        ("kf_racy_max", [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ["0", "1", "2", "3", "4"], "SafeSearch 'racy' (máx. nos quadros)"),
    ]
    fig, eixos = plt.subplots(2, 3, figsize=(15, 8.5))
    taxa_base = treino[COLUNA_ALVO].mean()
    tabelas = []
    for ax, (coluna, faixas, rotulos, titulo) in zip(eixos.ravel(), configuracoes):
        tabela = taxa_viral_por_faixa(treino, coluna, faixas, rotulos)
        tabelas.append(tabela.assign(variavel=coluna).reset_index(names="faixa"))
        ax.bar(tabela.index.astype(str), tabela["taxa_viral"], yerr=tabela["ic95"], color="#4C72B0", capsize=3)
        ax.axhline(taxa_base, color="#C44E52", ls="--", lw=1.5)
        for i, n in enumerate(tabela["n"]):
            ax.text(i, 0.005, f"n={n}", ha="center", va="bottom", fontsize=8, color="white")
        ax.set_title(titulo)
        ax.set_ylabel("taxa de viralização")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Taxa de viralização por faixa (treino) — linha tracejada = taxa base de 20%", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "02_taxa_viral_por_faixa.png", dpi=130)
    plt.close(fig)
    pd.concat(tabelas).to_csv(TABELAS_DIR / "taxa_viral_por_faixa.csv", index=False)


def grafico_correlacoes(treino):
    colunas = COLUNAS_CRIADOR + COLUNAS_TEXTO_CONTAGEM + COLUNAS_TEXTO_OUTRAS + [
        c for c in COLUNAS_VISUAIS if c.endswith("_media")
    ]
    base = treino[colunas].assign(log_visualizacoes=np.log1p(treino["visualizacoes"]))
    correlacao = base.corr(method="spearman")["log_visualizacoes"].drop("log_visualizacoes")
    correlacao = correlacao.sort_values()
    correlacao.to_frame("spearman").to_csv(TABELAS_DIR / "correlacao_spearman.csv")
    fig, ax = plt.subplots(figsize=(8, 7))
    cores = ["#C44E52" if v < 0 else "#4C72B0" for v in correlacao]
    ax.barh(correlacao.index, correlacao.values, color=cores)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("correlação de Spearman com log(visualizações)")
    ax.set_title("Associação monotônica de cada feature com visualizações (treino)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "03_correlacao_spearman.png", dpi=130)
    plt.close(fig)
    return correlacao


def tabela_hashtags(treino, suporte_minimo=40):
    explodido = treino[["hashtags_texto", COLUNA_ALVO]].assign(
        hashtag=treino["hashtags_texto"].str.split()
    ).explode("hashtag").dropna(subset=["hashtag"])
    explodido = explodido.drop_duplicates()
    agrupado = explodido.groupby("hashtag")[COLUNA_ALVO].agg(["mean", "size"])
    agrupado = agrupado[agrupado["size"] >= suporte_minimo].rename(columns={"mean": "taxa_viral", "size": "n"})
    agrupado["ic95"] = 1.96 * np.sqrt(agrupado["taxa_viral"] * (1 - agrupado["taxa_viral"]) / agrupado["n"])
    agrupado = agrupado.sort_values("taxa_viral", ascending=False)
    agrupado.to_csv(TABELAS_DIR / "hashtags_taxa_viral.csv")
    return agrupado


def main():
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    TABELAS_DIR.mkdir(parents=True, exist_ok=True)

    completude = comparar_completude()
    print("=== Completude por plataforma ===")
    print(completude.to_string(index=False))

    treino, teste, limiar = carregar_divisao()
    print(f"\nTreino: {len(treino)} | Teste: {len(teste)} | limiar viral (P80 treino): {limiar:,.0f} visualizações")
    print(f"Taxa viral treino: {treino[COLUNA_ALVO].mean():.3f} | teste: {teste[COLUNA_ALVO].mean():.3f}")

    grafico_distribuicao(treino, limiar)
    grafico_taxas(treino)
    correlacao = grafico_correlacoes(treino)
    print("\n=== Spearman com log(visualizações) ===")
    print(correlacao.round(3).to_string())

    hashtags = tabela_hashtags(treino)
    print("\n=== Hashtags com maior e menor taxa viral (n >= 40) ===")
    print(hashtags.head(12).round(3).to_string())
    print(hashtags.tail(8).round(3).to_string())

    resumo = {
        "n_treino": len(treino),
        "n_teste": len(teste),
        "limiar_viral": limiar,
        "taxa_viral_treino": treino[COLUNA_ALVO].mean(),
        "taxa_viral_teste": teste[COLUNA_ALVO].mean(),
        "pct_autor_ausente_treino": treino["seguidores"].isna().mean(),
    }
    pd.Series(resumo).to_csv(TABELAS_DIR / "resumo_divisao.csv", header=["valor"])


if __name__ == "__main__":
    main()
