import json

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import COLUNA_ALVO, MODELOS_DIR, N_FOLDS, RANDOM_STATE, TABELAS_DIR
from src.divisao import carregar_divisao
from src.modelos import catalogo_modelos, colunas_entrada, pipeline_para
from src.treinar_comparar import limiar_f1_oof


def treinar():
    comparacao = pd.read_csv(TABELAS_DIR / "comparacao_modelos.csv")
    linha = comparacao[comparacao["escolhido"]].iloc[0]
    nome = linha["modelo"]
    catalogo = catalogo_modelos()
    if nome not in catalogo:
        raise SystemExit(f"{nome} é um ensemble composto; rode python -m src.treinar_comparar")

    parametros = {k: v for k, v in json.loads(linha["hiperparametros"]).items() if not k.startswith("_")}
    modelo = pipeline_para(catalogo[nome]["estimador"])
    modelo.set_params(**{f"modelo__{k}": v for k, v in parametros.items()})

    treino, _, limiar_viral = carregar_divisao()
    colunas = colunas_entrada()
    X, y = treino[colunas], treino[COLUNA_ALVO]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    print(f"Treinando {nome} com {parametros}...", flush=True)
    limiar, _, _ = limiar_f1_oof(modelo, X, y, cv)
    modelo.fit(X, y)

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, MODELOS_DIR / "modelo_escolhido.joblib")
    metadados_existentes = {}
    caminho_json = MODELOS_DIR / "modelo_escolhido.json"
    if caminho_json.exists():
        metadados_existentes = json.loads(caminho_json.read_text())
    metadados_existentes.update(
        {
            "modelo": nome,
            "limiar_viral_visualizacoes": float(limiar_viral),
            "limiar_decisao_f1": limiar,
            "colunas": colunas,
            "taxa_viral_treino": float(y.mean()),
        }
    )
    caminho_json.write_text(json.dumps(metadados_existentes, indent=2, ensure_ascii=False))
    print(f"Modelo salvo em {MODELOS_DIR / 'modelo_escolhido.joblib'} (limiar de decisão {limiar:.4f})")


if __name__ == "__main__":
    treinar()
