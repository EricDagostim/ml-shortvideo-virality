import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.config import CATEGORIAS_SAFESEARCH, HASHTAGS_GENERICAS, MODELOS_DIR, TABELAS_DIR
from src.divisao import carregar_divisao
from src.features import features_texto
from src.modelos import pontuacoes


def montar_linha(legenda, seguidores, videos_autor, curtidas_autor, visuais):
    texto = features_texto(pd.Series([legenda]))
    criador = pd.DataFrame(
        {
            "seguidores": [seguidores],
            "curtidas_autor_outros": [curtidas_autor],
            "videos_autor": [videos_autor],
            "curtidas_por_video_autor": [
                curtidas_autor / max(videos_autor, 1) if curtidas_autor is not None and videos_autor is not None else np.nan
            ],
        }
    ).astype(float)
    visual = pd.DataFrame([visuais])
    return pd.concat([criador, texto, visual], axis=1)


def variantes(legenda, hashtags_sugeridas):
    tags = [t for t in legenda.split() if t.startswith("#")]
    texto = " ".join(t for t in legenda.split() if not t.startswith("#"))
    genericas = set(HASHTAGS_GENERICAS)
    cenarios = {"legenda original": legenda}
    sem_genericas = [t for t in tags if t.lower() not in genericas]
    if len(sem_genericas) != len(tags):
        cenarios["sem hashtags genéricas (#fyp, #viral...)"] = " ".join([texto] + sem_genericas)
    if len(tags) > 3:
        cenarios["só as 3 primeiras hashtags"] = " ".join([texto] + tags[:3])
    if tags:
        cenarios["sem nenhuma hashtag"] = texto
    if texto and not texto.rstrip().endswith("?"):
        cenarios["texto terminando em pergunta"] = " ".join([texto.rstrip(".! ") + "?"] + tags)
    if texto and texto.upper() == texto and any(c.isalpha() for c in texto):
        cenarios["texto sem CAIXA ALTA"] = " ".join([texto.capitalize()] + tags)
    if len(texto) > 80:
        cenarios["texto encurtado (60 caracteres)"] = " ".join([texto[:60]] + tags)
    for hashtag in hashtags_sugeridas:
        if hashtag not in [t.lower() for t in tags]:
            cenarios[f"adicionar {hashtag}"] = f"{legenda} {hashtag}".strip()
    return cenarios


def main():
    parser = argparse.ArgumentParser(description="Estima a chance de um vídeo de TikTok viralizar e sugere variações da legenda.")
    parser.add_argument("--legenda", required=True)
    parser.add_argument("--seguidores", type=float, default=None)
    parser.add_argument("--videos-autor", type=float, default=None)
    parser.add_argument("--curtidas-autor", type=float, default=None, help="total de curtidas do perfil")
    for categoria in CATEGORIAS_SAFESEARCH:
        parser.add_argument(f"--{categoria}", type=float, default=None, help="SafeSearch 0-4 (opcional)")
    parser.add_argument("--top-hashtags", type=int, default=5)
    args = parser.parse_args()

    modelo = joblib.load(MODELOS_DIR / "modelo_escolhido.joblib")
    metadados = json.loads((MODELOS_DIR / "modelo_escolhido.json").read_text())
    treino, _, limiar = carregar_divisao()
    colunas = metadados["colunas"]

    visuais = {}
    for categoria in CATEGORIAS_SAFESEARCH:
        valor = getattr(args, categoria)
        for agregacao in ("media", "max"):
            coluna = f"kf_{categoria}_{agregacao}"
            visuais[coluna] = valor if valor is not None else float(treino[coluna].median())

    sugeridas = []
    caminho_efeitos = TABELAS_DIR / "efeito_hashtags.csv"
    if caminho_efeitos.exists():
        efeitos = pd.read_csv(caminho_efeitos)
        sugeridas = efeitos[efeitos["delta_medio"] > 0]["hashtag"].head(args.top_hashtags).tolist()

    referencia = pontuacoes(modelo, treino[colunas])
    probabilistico = hasattr(modelo, "predict_proba")

    linhas = []
    for nome, legenda in variantes(args.legenda, sugeridas).items():
        X = montar_linha(legenda, args.seguidores, args.videos_autor, args.curtidas_autor, visuais)[colunas]
        escore = float(pontuacoes(modelo, X)[0])
        linhas.append(
            {
                "cenario": nome,
                "escore": escore,
                "percentil_vs_treino": 100 * float(np.mean(referencia <= escore)),
                "previsto_viral": escore >= metadados["limiar_decisao_f1"],
            }
        )
    resultado = pd.DataFrame(linhas)
    base = resultado.iloc[0]["escore"]
    resultado["delta"] = resultado["escore"] - base
    resultado = pd.concat([resultado.iloc[:1], resultado.iloc[1:].sort_values("delta", ascending=False)])

    rotulo = "P(viral)" if probabilistico else "escore"
    print(f"Modelo: {metadados['modelo']} | viral = top 20% (>= {limiar:,.0f} visualizações)")
    print(f"Taxa base de viralização: {metadados['taxa_viral_treino']:.1%}\n")
    exibicao = resultado.rename(columns={"escore": rotulo})
    print(exibicao.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "\nAtenção: as variações mostram associações aprendidas em dados observacionais; "
        "não garantem causa e efeito. Teste as melhores ideias na prática (ex.: A/B entre vídeos)."
    )


if __name__ == "__main__":
    main()
