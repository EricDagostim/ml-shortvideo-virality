import json
import time

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from src.config import (
    COLUNA_ALVO,
    FIGURAS_DIR,
    MODELOS_DIR,
    N_BOOTSTRAP,
    N_FOLDS,
    RANDOM_STATE,
    TABELAS_DIR,
)
from src.divisao import carregar_divisao
from src.modelos import catalogo_modelos, colunas_entrada, montar_ensembles, pipeline_para, pontuacoes

sns.set_theme(style="whitegrid", context="notebook")


def ajustar(nome, especificacao, X, y, cv):
    pipeline = pipeline_para(especificacao["estimador"])
    inicio = time.time()
    espaco = especificacao.get("espaco")
    if espaco is None:
        resultado = cross_validate(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipeline.fit(X, y)
        melhor = pipeline
        auc_folds = resultado["test_score"]
        parametros = {}
        tempo_fit = float(np.mean(resultado["fit_time"]))
    else:
        n_combinacoes = int(np.prod([len(v) for v in espaco.values()]))
        n_iter = especificacao.get("n_iter")
        if n_iter is None or n_iter >= n_combinacoes:
            busca = GridSearchCV(pipeline, espaco, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
        else:
            busca = RandomizedSearchCV(
                pipeline, espaco, n_iter=n_iter, scoring="roc_auc", cv=cv,
                n_jobs=-1, refit=True, random_state=RANDOM_STATE,
            )
        busca.fit(X, y)
        melhor = busca.best_estimator_
        indice = busca.best_index_
        auc_folds = np.array([busca.cv_results_[f"split{k}_test_score"][indice] for k in range(cv.get_n_splits())])
        parametros = {k.replace("modelo__", ""): v for k, v in busca.best_params_.items()}
        tempo_fit = float(busca.cv_results_["mean_fit_time"][indice])
        bordas = [
            k.replace("modelo__", "")
            for k, valores in espaco.items()
            if len(valores) > 2
            and all(isinstance(v, (int, float)) for v in valores)
            and busca.best_params_[k] in (min(valores), max(valores))
        ]
        parametros["_na_borda"] = bordas
    print(
        f"[{nome}] AUC CV = {auc_folds.mean():.4f} ± {auc_folds.std():.4f} "
        f"| {time.time() - inicio:.0f}s | {parametros}",
        flush=True,
    )
    return melhor, auc_folds, parametros, tempo_fit


def limiar_f1_oof(modelo, X, y, cv):
    metodo = "predict_proba" if hasattr(modelo, "predict_proba") else "decision_function"
    oof = cross_val_predict(modelo, X, y, cv=cv, method=metodo, n_jobs=-1)
    if oof.ndim == 2:
        oof = oof[:, 1]
    candidatos = np.unique(np.quantile(oof, np.linspace(0.30, 0.95, 131)))
    f1s = [f1_score(y, (oof >= c).astype(int)) for c in candidatos]
    melhor = int(np.argmax(f1s))
    return float(candidatos[melhor]), float(f1s[melhor]), oof


def avaliar_teste(modelo, limiar, X, y):
    escores = pontuacoes(modelo, X)
    previsto = (escores >= limiar).astype(int)
    metricas = {
        "auc_teste": roc_auc_score(y, escores),
        "f1_teste": f1_score(y, previsto),
        "precision_teste": precision_score(y, previsto, zero_division=0),
        "recall_teste": recall_score(y, previsto),
        "accuracy_teste": accuracy_score(y, previsto),
        "log_loss_teste": log_loss(y, np.clip(escores, 1e-6, 1 - 1e-6)) if hasattr(modelo, "predict_proba") else np.nan,
    }
    return metricas, escores


def bootstrap_auc(y, escores_por_modelo, referencia, n=N_BOOTSTRAP):
    gerador = np.random.default_rng(RANDOM_STATE)
    y = np.asarray(y)
    nomes = list(escores_por_modelo)
    aucs = {nome: [] for nome in nomes}
    for _ in range(n):
        indices = gerador.integers(0, len(y), len(y))
        if y[indices].min() == y[indices].max():
            continue
        for nome in nomes:
            aucs[nome].append(roc_auc_score(y[indices], escores_por_modelo[nome][indices]))
    aucs = {k: np.array(v) for k, v in aucs.items()}
    linhas = []
    for nome in nomes:
        diferenca = aucs[referencia] - aucs[nome]
        linhas.append(
            {
                "modelo": nome,
                "auc_teste_ic95_inf": np.percentile(aucs[nome], 2.5),
                "auc_teste_ic95_sup": np.percentile(aucs[nome], 97.5),
                "dif_auc_vs_escolhido": diferenca.mean(),
                "dif_ic95_inf": np.percentile(diferenca, 2.5),
                "dif_ic95_sup": np.percentile(diferenca, 97.5),
                "p_escolhido_nao_melhor": float(np.mean(diferenca <= 0)),
            }
        )
    return pd.DataFrame(linhas)


def escolher_modelo(tabela):
    candidatos = tabela[tabela["modelo"] != "Dummy (baseline)"]
    melhor = candidatos.loc[candidatos["auc_cv_media"].idxmax()]
    erro_padrao = melhor["auc_cv_desvio"] / np.sqrt(N_FOLDS)
    limite = melhor["auc_cv_media"] - erro_padrao
    empatados = candidatos[candidatos["auc_cv_media"] >= limite]
    escolhido = empatados.sort_values(["complexidade", "auc_cv_media"], ascending=[True, False]).iloc[0]
    return melhor["modelo"], escolhido["modelo"], limite, empatados["modelo"].tolist()


def graficos(tabela, escores_teste, y_teste, escolhido, limiar_escolhido, bootstrap):
    ordem = tabela.sort_values("auc_cv_media")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    cores = ["#C44E52" if m == escolhido else "#4C72B0" for m in ordem["modelo"]]
    ax.barh(ordem["modelo"], ordem["auc_cv_media"], xerr=ordem["auc_cv_desvio"], color=cores, capsize=3)
    ax.scatter(ordem["auc_teste"], ordem["modelo"], color="black", marker="D", zorder=3, label="AUC no teste")
    ax.set_xlim(0.45, max(0.8, ordem["auc_teste"].max() + 0.03))
    ax.axvline(0.5, color="gray", ls=":")
    ax.set_xlabel("ROC-AUC (barra = média CV 5 folds ± desvio; losango = teste)")
    ax.set_title("Comparação dos modelos — TikTok, features pré-publicação")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "04_comparacao_auc.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 7))
    destaque = tabela.sort_values("auc_cv_media", ascending=False)["modelo"].head(6).tolist()
    for nome in destaque + ["Dummy (baseline)"]:
        fpr, tpr, _ = roc_curve(y_teste, escores_teste[nome])
        largura = 2.8 if nome == escolhido else 1.3
        ax.plot(fpr, tpr, lw=largura, label=f"{nome} (AUC={roc_auc_score(y_teste, escores_teste[nome]):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":")
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (recall)")
    ax.set_title("Curvas ROC no conjunto de teste (top 6 da CV + baseline)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "05_curvas_roc_teste.png", dpi=130)
    plt.close(fig)

    previsto = (escores_teste[escolhido] >= limiar_escolhido).astype(int)
    matriz = confusion_matrix(y_teste, previsto)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["não viral", "viral"], yticklabels=["não viral", "viral"])
    ax.set_xlabel("previsto")
    ax.set_ylabel("real")
    ax.set_title(f"Matriz de confusão — {escolhido} (teste)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "06_matriz_confusao_escolhido.png", dpi=130)
    plt.close(fig)

    dados = bootstrap[bootstrap["modelo"] != escolhido].sort_values("dif_auc_vs_escolhido")
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.errorbar(
        dados["dif_auc_vs_escolhido"], dados["modelo"],
        xerr=[dados["dif_auc_vs_escolhido"] - dados["dif_ic95_inf"], dados["dif_ic95_sup"] - dados["dif_auc_vs_escolhido"]],
        fmt="o", color="#4C72B0", capsize=3,
    )
    ax.axvline(0, color="#C44E52", ls="--")
    ax.set_xlabel(f"AUC({escolhido}) − AUC(modelo) no teste — IC 95% bootstrap pareado")
    ax.set_title("O modelo escolhido é melhor? (IC à direita de 0 = sim)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / "07_bootstrap_diferencas_auc.png", dpi=130)
    plt.close(fig)


def main():
    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    treino, teste, limiar_viral = carregar_divisao()
    colunas = colunas_entrada()
    X_treino, y_treino = treino[colunas], treino[COLUNA_ALVO]
    X_teste, y_teste = teste[colunas], teste[COLUNA_ALVO]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Treino {X_treino.shape} | Teste {X_teste.shape} | limiar viral {limiar_viral:,.0f}", flush=True)

    catalogo = catalogo_modelos()
    melhores, linhas = {}, []
    for etapa in ("individuais", "ensembles"):
        if etapa == "ensembles":
            catalogo = montar_ensembles(melhores)
        for nome, especificacao in catalogo.items():
            modelo, auc_folds, parametros, tempo_fit = ajustar(nome, especificacao, X_treino, y_treino, cv)
            melhores[nome] = modelo
            linhas.append(
                {
                    "modelo": nome,
                    "auc_cv_media": auc_folds.mean(),
                    "auc_cv_desvio": auc_folds.std(),
                    "auc_cv_folds": json.dumps([round(float(a), 4) for a in auc_folds]),
                    "tempo_fit_s": tempo_fit,
                    "complexidade": especificacao["complexidade"],
                    "hiperparametros": json.dumps(parametros, default=str, ensure_ascii=False),
                }
            )

    tabela = pd.DataFrame(linhas)
    melhor_cv, escolhido, limite_1ep, empatados = escolher_modelo(tabela)
    print(f"\nMaior AUC CV: {melhor_cv} | limite 1-EP: {limite_1ep:.4f} | empatados: {empatados}")
    print(f"Modelo escolhido pela regra de 1 erro-padrão: {escolhido}", flush=True)

    escores_teste, limiares, extras = {}, {}, []
    for nome, modelo in melhores.items():
        limiar, f1_oof, _ = limiar_f1_oof(modelo, X_treino, y_treino, cv)
        metricas, escores = avaliar_teste(modelo, limiar, X_teste, y_teste)
        inicio = time.time()
        pontuacoes(modelo, X_teste)
        metricas["tempo_predicao_teste_s"] = time.time() - inicio
        escores_teste[nome] = escores
        limiares[nome] = limiar
        extras.append({"modelo": nome, "limiar_f1_oof": limiar, "f1_cv_oof": f1_oof, **metricas})
        joblib.dump(modelo, MODELOS_DIR / f"{nome.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

    tabela = tabela.merge(pd.DataFrame(extras), on="modelo")
    bootstrap = bootstrap_auc(y_teste, escores_teste, escolhido)
    tabela = tabela.merge(bootstrap, on="modelo").sort_values("auc_cv_media", ascending=False)
    tabela["escolhido"] = tabela["modelo"] == escolhido
    tabela.to_csv(TABELAS_DIR / "comparacao_modelos.csv", index=False)

    colunas_exibir = [
        "modelo", "auc_cv_media", "auc_cv_desvio", "auc_teste", "auc_teste_ic95_inf", "auc_teste_ic95_sup",
        "f1_teste", "precision_teste", "recall_teste", "accuracy_teste", "log_loss_teste",
        "dif_auc_vs_escolhido", "dif_ic95_inf", "dif_ic95_sup", "tempo_fit_s",
    ]
    print("\n" + tabela[colunas_exibir].round(4).to_string(index=False))

    graficos(tabela, escores_teste, y_teste, escolhido, limiares[escolhido], bootstrap)

    joblib.dump(melhores[escolhido], MODELOS_DIR / "modelo_escolhido.joblib")
    metadados = {
        "modelo": escolhido,
        "maior_auc_cv": melhor_cv,
        "limite_1ep": limite_1ep,
        "empatados_1ep": empatados,
        "limiar_viral_visualizacoes": float(limiar_viral),
        "limiar_decisao_f1": limiares[escolhido],
        "colunas": colunas,
        "taxa_viral_treino": float(y_treino.mean()),
    }
    (MODELOS_DIR / "modelo_escolhido.json").write_text(json.dumps(metadados, indent=2, ensure_ascii=False))
    pd.Series(metadados, dtype=object).to_frame("valor").to_csv(TABELAS_DIR / "modelo_escolhido.csv")


if __name__ == "__main__":
    main()
