import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.config import COLUNA_ALVO, FIGURAS_DIR, RANDOM_STATE, TABELAS_DIR
from src.divisao import carregar_divisao

sns.set_theme(style="whitegrid", context="notebook")

COLUNAS_LOG = ["comprimento_legenda", "n_hashtags", "n_hashtags_genericas", "n_mencoes", "n_emojis"]
COLUNAS_DIRETAS = [
    "tem_pergunta", "tem_exclamacao", "prop_maiusculas", "tem_acento_pt", "alfabeto_arabe",
    "kf_adult_media", "kf_medical_media", "kf_spoofed_media", "kf_violence_media", "kf_racy_media",
]
COLUNAS = COLUNAS_LOG + COLUNAS_DIRETAS


def preparacao():
    return ColumnTransformer(
        [
            ("log", Pipeline([
                ("log", FunctionTransformer(np.log1p)),
                ("imputador", SimpleImputer(strategy="median")),
                ("escala", StandardScaler()),
            ]), COLUNAS_LOG),
            ("diretas", Pipeline([
                ("imputador", SimpleImputer(strategy="median")),
                ("escala", StandardScaler()),
            ]), COLUNAS_DIRETAS),
        ]
    )


def avaliar_k(Z, ks):
    linhas = []
    for k in ks:
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE).fit(Z)
        silhueta = silhouette_score(Z, modelo.labels_, sample_size=5000, random_state=RANDOM_STATE)
        linhas.append({"k": k, "inercia": modelo.inertia_, "silhueta": silhueta})
        print(f"K={k}: inércia={modelo.inertia_:,.0f} silhueta={silhueta:.3f}", flush=True)
    return pd.DataFrame(linhas)


def estabilidade(Z, k, referencia):
    gerador = np.random.default_rng(RANDOM_STATE)
    ari_sementes = []
    for semente in range(1, 11):
        rotulos = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=semente).fit_predict(Z)
        ari_sementes.append(adjusted_rand_score(referencia, rotulos))
    ari_subamostras = []
    for _ in range(20):
        indices = gerador.choice(len(Z), size=int(0.8 * len(Z)), replace=False)
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE).fit(Z[indices])
        ari_subamostras.append(adjusted_rand_score(referencia, modelo.predict(Z)))
    return pd.DataFrame(
        [
            {"teste": "10 sementes (n_init=20)", "ari_min": np.min(ari_sementes), "ari_mediano": np.median(ari_sementes)},
            {"teste": "20 subamostras de 80%", "ari_min": np.min(ari_subamostras), "ari_mediano": np.median(ari_subamostras)},
        ]
    )


def perfis(treino, teste, rotulos_treino, rotulos_teste):
    base = treino.assign(grupo=rotulos_treino)
    resumo = base.groupby("grupo")[COLUNAS].median()
    taxa = base.groupby("grupo")[COLUNA_ALVO].agg(taxa_viral_treino="mean", n_treino="size")
    taxa["ic95_treino"] = 1.96 * np.sqrt(taxa["taxa_viral_treino"] * (1 - taxa["taxa_viral_treino"]) / taxa["n_treino"])
    taxa_teste = teste.assign(grupo=rotulos_teste).groupby("grupo")[COLUNA_ALVO].agg(taxa_viral_teste="mean", n_teste="size")
    seguidores = base.groupby("grupo")["seguidores"].median().rename("seguidores_mediana")
    return pd.concat([taxa, taxa_teste, seguidores, resumo], axis=1)


def separacao_por_variavel(Z, rotulos, nomes):
    linhas = []
    for j, nome in enumerate(nomes):
        total = np.var(Z[:, j]) * len(Z)
        dentro = sum(np.var(Z[rotulos == g, j]) * np.sum(rotulos == g) for g in np.unique(rotulos))
        linhas.append({"variavel": nome, "variacao_entre_grupos_pct": 100 * (1 - dentro / total)})
    return pd.DataFrame(linhas).sort_values("variacao_entre_grupos_pct", ascending=False)


def graficos(avaliacao, Z, rotulos, k, tabela_perfis):
    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
    eixos[0].plot(avaliacao["k"], avaliacao["inercia"], "o-")
    eixos[0].set_title("Método do cotovelo (inércia)")
    eixos[0].set_xlabel("K")
    eixos[1].plot(avaliacao["k"], avaliacao["silhueta"], "o-", color="#C44E52")
    eixos[1].axvline(k, color="gray", ls="--")
    eixos[1].set_title("Silhueta média")
    eixos[1].set_xlabel("K")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "12_kmeans_escolha_k.png", dpi=130)
    plt.close(fig)

    gerador = np.random.default_rng(RANDOM_STATE)
    amostra = gerador.choice(len(Z), size=min(4000, len(Z)), replace=False)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(Z)
    projecao = pca.transform(Z[amostra])
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.scatterplot(x=projecao[:, 0], y=projecao[:, 1], hue=rotulos[amostra], palette="tab10", s=12, alpha=0.6, ax=eixos[0], legend="full")
    eixos[0].set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% da variância)")
    eixos[0].set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    eixos[0].set_title(f"Perfis de conteúdo (K={k}) projetados por PCA")
    dados = tabela_perfis.reset_index(names="grupo")
    eixos[1].bar(dados["grupo"].astype(str), dados["taxa_viral_treino"], yerr=dados["ic95_treino"], capsize=4, color="#4C72B0", label="treino")
    eixos[1].scatter(dados["grupo"].astype(str), dados["taxa_viral_teste"], color="black", marker="D", zorder=3, label="teste")
    eixos[1].axhline(0.2, color="#C44E52", ls="--")
    eixos[1].set_title("Taxa de viralização por perfil de conteúdo")
    eixos[1].set_xlabel("grupo")
    eixos[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "13_kmeans_perfis.png", dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    treino, teste, _ = carregar_divisao()
    prep = preparacao()
    Z = prep.fit_transform(treino[COLUNAS])
    Z_teste = prep.transform(teste[COLUNAS])

    avaliacao = avaliar_k(Z, range(2, 11))
    avaliacao.to_csv(TABELAS_DIR / "kmeans_avaliacao_k.csv", index=False)
    k = args.k or int(avaliacao.loc[avaliacao["silhueta"].idxmax(), "k"])
    print(f"K escolhido: {k}")

    modelo = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE).fit(Z)
    rotulos = modelo.labels_
    rotulos_teste = modelo.predict(Z_teste)

    tabela_estabilidade = estabilidade(Z, k, rotulos)
    tabela_estabilidade.to_csv(TABELAS_DIR / "kmeans_estabilidade.csv", index=False)
    print(tabela_estabilidade.round(3).to_string(index=False))

    tabela_perfis = perfis(treino, teste, rotulos, rotulos_teste)
    tabela_perfis.to_csv(TABELAS_DIR / "kmeans_perfis.csv")
    print(tabela_perfis.round(3).T.to_string())

    separacao = separacao_por_variavel(Z, rotulos, COLUNAS)
    separacao.to_csv(TABELAS_DIR / "kmeans_separacao_variaveis.csv", index=False)
    print(separacao.round(1).to_string(index=False))

    graficos(avaliacao, Z, rotulos, k, tabela_perfis)


if __name__ == "__main__":
    main()
