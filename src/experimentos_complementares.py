import json

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    COLUNA_ALVO,
    COLUNA_ALVO_CONTINUO,
    FIGURAS_DIR,
    MODELOS_DIR,
    N_FOLDS,
    QUANTIL_VIRAL,
    RANDOM_STATE,
    TABELAS_DIR,
)
from src.divisao import carregar_divisao
from src.features import montar_tiktok, rotular_viral
from src.modelos import colunas_entrada, pipeline_para, pontuacoes

sns.set_theme(style="whitegrid", context="notebook")


def avaliar(estimador, treino, teste, cv, **kwargs_prep):
    colunas = colunas_entrada(**kwargs_prep)
    pipeline = pipeline_para(estimador, **kwargs_prep)
    resultado = cross_validate(pipeline, treino[colunas], treino[COLUNA_ALVO], cv=cv, scoring="roc_auc", n_jobs=-1)
    pipeline.fit(treino[colunas], treino[COLUNA_ALVO])
    auc_teste = roc_auc_score(teste[COLUNA_ALVO], pontuacoes(pipeline, teste[colunas]))
    return resultado["test_score"].mean(), resultado["test_score"].std(), auc_teste


def experimento_vazamento(treino, teste, cv):
    modelos = {
        "Regressão Logística": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1),
    }
    linhas = []
    for nome, estimador in modelos.items():
        for cenario, kwargs in [
            ("pré-publicação (usado)", {}),
            ("+ curtidas/comentários/compart./salvamentos (vazamento)", {"engajamento": True}),
        ]:
            media, desvio, auc_teste = avaliar(estimador, treino, teste, cv, **kwargs)
            linhas.append({"modelo": nome, "cenario": cenario, "auc_cv_media": media, "auc_cv_desvio": desvio, "auc_teste": auc_teste})
            print(f"[vazamento] {nome} | {cenario}: CV {media:.4f} ± {desvio:.4f} | teste {auc_teste:.4f}", flush=True)
    return pd.DataFrame(linhas)


def experimento_ablacao(estimador, treino, teste, cv):
    cenarios = [
        ("só criador", {"criador": True, "conteudo": False}),
        ("só conteúdo", {"criador": False, "conteudo": True}),
        ("criador + conteúdo", {"criador": True, "conteudo": True}),
    ]
    linhas = []
    for cenario, kwargs in cenarios:
        media, desvio, auc_teste = avaliar(estimador, treino, teste, cv, **kwargs)
        linhas.append({"cenario": cenario, "auc_cv_media": media, "auc_cv_desvio": desvio, "auc_teste": auc_teste})
        print(f"[ablação] {cenario}: CV {media:.4f} ± {desvio:.4f} | teste {auc_teste:.4f}", flush=True)

    com_autor = treino["seguidores"].notna()
    teste_com_autor = teste["seguidores"].notna()
    for cenario, kwargs in cenarios[1:]:
        media, desvio, auc_teste = avaliar(estimador, treino[com_autor], teste[teste_com_autor], cv, **kwargs)
        linhas.append({"cenario": f"{cenario} (só vídeos com dados do criador)", "auc_cv_media": media, "auc_cv_desvio": desvio, "auc_teste": auc_teste})
        print(f"[ablação] {cenario} (com autor): CV {media:.4f} ± {desvio:.4f} | teste {auc_teste:.4f}", flush=True)
    return pd.DataFrame(linhas)


def experimento_temporal(estimador):
    df = montar_tiktok().sort_values("dia_coleta")
    dias = df["dia_coleta"].drop_duplicates().sort_values().reset_index(drop=True)
    acumulado = df.groupby("dia_coleta").size().cumsum() / len(df)
    corte = acumulado[acumulado >= 1 - 0.20].index[0]
    treino = df[df["dia_coleta"] < corte].copy()
    teste = df[df["dia_coleta"] >= corte].copy()
    limiar = treino[COLUNA_ALVO_CONTINUO].quantile(QUANTIL_VIRAL)
    treino[COLUNA_ALVO] = rotular_viral(treino[COLUNA_ALVO_CONTINUO], limiar)
    teste[COLUNA_ALVO] = rotular_viral(teste[COLUNA_ALVO_CONTINUO], limiar)
    colunas = colunas_entrada()
    pipeline = pipeline_para(estimador)
    pipeline.fit(treino[colunas], treino[COLUNA_ALVO])
    auc = roc_auc_score(teste[COLUNA_ALVO], pontuacoes(pipeline, teste[colunas]))
    resultado = {
        "treino_ate": str((corte - pd.Timedelta(days=1)).date()),
        "teste_a_partir_de": str(corte.date()),
        "n_treino": len(treino),
        "n_teste": len(teste),
        "dias_coleta": len(dias),
        "taxa_viral_teste": teste[COLUNA_ALVO].mean(),
        "auc_teste_temporal": auc,
    }
    print(f"[temporal] {resultado}", flush=True)
    return pd.DataFrame([resultado])


def grafico(vazamento, ablacao, temporal, auc_aleatorio):
    fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
    ax = eixos[0]
    tabela_pivo = vazamento.pivot(index="modelo", columns="cenario", values="auc_teste")
    tabela_pivo.plot.barh(ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("ROC-AUC no teste")
    ax.set_title("Vazamento: métricas pós-publicação tornam o problema 'trivial'")
    ax.legend(fontsize=8, loc="lower right")

    ax = eixos[1]
    dados = pd.concat(
        [
            ablacao[["cenario", "auc_teste"]],
            pd.DataFrame([
                {"cenario": "criador + conteúdo — divisão aleatória", "auc_teste": auc_aleatorio},
                {"cenario": "criador + conteúdo — divisão temporal", "auc_teste": temporal["auc_teste_temporal"].iloc[0]},
            ]),
        ]
    )
    ax.barh(dados["cenario"], dados["auc_teste"], color="#4C72B0")
    for i, v in enumerate(dados["auc_teste"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, max(0.8, dados["auc_teste"].max() + 0.05))
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlabel("ROC-AUC no teste")
    ax.set_title("Ablação de grupos de features e robustez temporal")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "08_vazamento_ablacao_temporal.png", dpi=130)
    plt.close(fig)


def main():
    treino, teste, _ = carregar_divisao()
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    escolhido = joblib.load(MODELOS_DIR / "modelo_escolhido.joblib")
    metadados = json.loads((MODELOS_DIR / "modelo_escolhido.json").read_text())
    estimador = clone(escolhido.named_steps["modelo"])
    print(f"Modelo escolhido: {metadados['modelo']}", flush=True)

    vazamento = experimento_vazamento(treino, teste, cv)
    ablacao = experimento_ablacao(estimador, treino, teste, cv)
    temporal = experimento_temporal(estimador)
    auc_aleatorio = roc_auc_score(teste[COLUNA_ALVO], pontuacoes(escolhido, teste[metadados["colunas"]]))

    vazamento.to_csv(TABELAS_DIR / "experimento_vazamento.csv", index=False)
    ablacao.to_csv(TABELAS_DIR / "experimento_ablacao.csv", index=False)
    temporal.to_csv(TABELAS_DIR / "experimento_temporal.csv", index=False)
    grafico(vazamento, ablacao, temporal, auc_aleatorio)


if __name__ == "__main__":
    main()
