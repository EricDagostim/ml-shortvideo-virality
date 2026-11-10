import json

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.config import COLUNA_ALVO, FIGURAS_DIR, HASHTAGS_GENERICAS, MODELOS_DIR, RANDOM_STATE, TABELAS_DIR
from src.divisao import carregar_divisao
from src.modelos import pontuacoes

sns.set_theme(style="whitegrid", context="notebook")

GRUPOS = {
    "seguidores": "criador", "curtidas_autor_outros": "criador", "videos_autor": "criador",
    "curtidas_por_video_autor": "criador", "hashtags_texto": "hashtags (texto)",
}


def grupo_da_coluna(coluna):
    if coluna in GRUPOS:
        return GRUPOS[coluna]
    if coluna.startswith("kf_"):
        return "visual (SafeSearch)"
    return "legenda"


def importancia_permutacao(modelo, X, y):
    resultado = permutation_importance(
        modelo, X, y, scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    tabela = pd.DataFrame(
        {
            "feature": X.columns,
            "queda_auc_media": resultado.importances_mean,
            "queda_auc_desvio": resultado.importances_std,
        }
    )
    tabela["grupo"] = tabela["feature"].map(grupo_da_coluna)
    return tabela.sort_values("queda_auc_media", ascending=False)


def importancia_impureza(modelo):
    estimador = modelo.named_steps["modelo"]
    if not hasattr(estimador, "feature_importances_"):
        return None
    nomes = modelo.named_steps["prep"].get_feature_names_out()
    return pd.DataFrame({"feature": nomes, "importancia_impureza": estimador.feature_importances_}).sort_values(
        "importancia_impureza", ascending=False
    )


def efeito_medio(modelo, X, coluna, valores):
    linhas = []
    for valor in valores:
        cenario = X.copy()
        cenario[coluna] = valor
        linhas.append({"feature": coluna, "valor": valor, "escore_medio": float(np.mean(pontuacoes(modelo, cenario)))})
    return linhas


def adicionar_hashtag(X, hashtag):
    cenario = X.copy()
    tem = cenario["hashtags_texto"].str.split().apply(lambda tags: hashtag in tags)
    cenario.loc[~tem, "hashtags_texto"] = (cenario.loc[~tem, "hashtags_texto"] + " " + hashtag).str.strip()
    cenario.loc[~tem, "n_hashtags"] += 1
    if hashtag in HASHTAGS_GENERICAS:
        cenario.loc[~tem, "n_hashtags_genericas"] += 1
    cenario.loc[~tem, "tem_legenda"] = 1
    return cenario, ~tem


def efeito_hashtags(modelo, X):
    vocabulario = modelo.named_steps["prep"].named_transformers_["hashtags"].get_feature_names_out()
    base = pontuacoes(modelo, X)
    linhas = []
    for hashtag in vocabulario:
        cenario, afetados = adicionar_hashtag(X, hashtag)
        if afetados.sum() == 0:
            continue
        novo = pontuacoes(modelo, cenario)
        linhas.append(
            {
                "hashtag": hashtag,
                "videos_afetados": int(afetados.sum()),
                "delta_medio": float(np.mean(novo[afetados.values] - base[afetados.values])),
            }
        )
    return pd.DataFrame(linhas).sort_values("delta_medio", ascending=False)


def graficos(permutacao, impureza, efeitos, hashtags, nome_modelo, rotulo_escore):
    top = permutacao.head(20).iloc[::-1]
    paleta = {"criador": "#C44E52", "legenda": "#4C72B0", "hashtags (texto)": "#55A868", "visual (SafeSearch)": "#8172B2"}
    paineis = 2 if impureza is not None else 1
    fig, eixos = plt.subplots(1, paineis, figsize=(8 * paineis, 7))
    eixos = np.atleast_1d(eixos)
    eixos[0].barh(top["feature"], top["queda_auc_media"], xerr=top["queda_auc_desvio"],
                  color=[paleta[g] for g in top["grupo"]], capsize=2)
    eixos[0].set_xlabel("queda média do ROC-AUC ao embaralhar (teste, 10 repetições)")
    eixos[0].set_title(f"Permutation importance — {nome_modelo}")
    for grupo, cor in paleta.items():
        eixos[0].bar(0, 0, color=cor, label=grupo)
    eixos[0].legend(loc="lower right", fontsize=8)
    if impureza is not None:
        top_imp = impureza.head(20).iloc[::-1]
        eixos[1].barh(top_imp["feature"], top_imp["importancia_impureza"], color="#937860")
        eixos[1].set_xlabel("importância por impureza (feature_importances_)")
        eixos[1].set_title("Importância interna do modelo (pode favorecer variáveis contínuas)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "09_importancia_features.png", dpi=130)
    plt.close(fig)

    titulos = {
        "seguidores": ("Seguidores do criador", True),
        "n_hashtags": ("Nº de hashtags", False),
        "n_hashtags_genericas": ("Hashtags genéricas", False),
        "comprimento_legenda": ("Tamanho da legenda (caracteres)", False),
        "n_emojis": ("Nº de emojis", False),
        "n_mencoes": ("Nº de menções (@)", False),
        "tem_pergunta": ("Legenda com pergunta (0/1)", False),
        "tem_acento_pt": ("Legenda em português (0/1)", False),
        "kf_racy_media": ("SafeSearch 'racy' médio", False),
    }
    fig, eixos = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (coluna, (titulo, log)) in zip(eixos.ravel(), titulos.items()):
        dados = efeitos[efeitos["feature"] == coluna]
        ax.plot(dados["valor"], dados["escore_medio"], "o-", color="#4C72B0")
        if log:
            ax.set_xscale("log")
        ax.set_title(titulo)
        ax.set_ylabel(rotulo_escore)
    fig.suptitle(f"Efeito médio estimado (dependência parcial manual) — {nome_modelo}, teste", y=1.0)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "10_efeito_medio_features.png", dpi=130)
    plt.close(fig)

    dados = hashtags.sort_values("delta_medio")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(dados["hashtag"], dados["delta_medio"], color=["#C44E52" if v < 0 else "#55A868" for v in dados["delta_medio"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(f"variação média em {rotulo_escore} ao ADICIONAR a hashtag")
    ax.set_title("E se eu adicionar esta hashtag? (vídeos de teste que não a tinham)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "11_efeito_hashtags.png", dpi=130)
    plt.close(fig)


def main():
    _, teste, _ = carregar_divisao()
    modelo = joblib.load(MODELOS_DIR / "modelo_escolhido.joblib")
    metadados = json.loads((MODELOS_DIR / "modelo_escolhido.json").read_text())
    X, y = teste[metadados["colunas"]], teste[COLUNA_ALVO]
    probabilistico = hasattr(modelo, "predict_proba")
    rotulo_escore = "P(viral) média" if probabilistico else "escore médio de decisão"

    permutacao = importancia_permutacao(modelo, X, y)
    permutacao.to_csv(TABELAS_DIR / "importancia_permutacao.csv", index=False)
    print(permutacao.head(20).round(4).to_string(index=False))
    print("\nPor grupo (soma das quedas de AUC):")
    print(permutacao.groupby("grupo")["queda_auc_media"].sum().sort_values(ascending=False).round(4).to_string())

    impureza = importancia_impureza(modelo)
    if impureza is not None:
        impureza.to_csv(TABELAS_DIR / "importancia_impureza.csv", index=False)

    grades = {
        "seguidores": [1e2, 1e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, 3e7],
        "n_hashtags": [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
        "n_hashtags_genericas": [0, 1, 2, 3, 4, 5],
        "comprimento_legenda": [0, 15, 30, 45, 60, 90, 120, 180, 250, 400],
        "n_emojis": [0, 1, 2, 3, 4, 6, 10],
        "n_mencoes": [0, 1, 2, 3, 5],
        "tem_pergunta": [0, 1],
        "tem_acento_pt": [0, 1],
        "kf_racy_media": [0, 0.5, 1, 1.5, 2, 2.5, 3, 4],
    }
    efeitos = pd.DataFrame([linha for coluna, valores in grades.items() for linha in efeito_medio(modelo, X, coluna, valores)])
    efeitos.to_csv(TABELAS_DIR / "efeito_medio_features.csv", index=False)

    hashtags = efeito_hashtags(modelo, X)
    hashtags.to_csv(TABELAS_DIR / "efeito_hashtags.csv", index=False)
    print("\nEfeito de adicionar hashtags:")
    print(hashtags.round(4).to_string(index=False))

    graficos(permutacao, impureza, efeitos, hashtags, metadados["modelo"], rotulo_escore)


if __name__ == "__main__":
    main()
