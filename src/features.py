import re

import numpy as np
import pandas as pd

from src.config import ARQUIVOS, CATEGORIAS_SAFESEARCH, DADOS_DIR, HASHTAGS_GENERICAS

PADRAO_HASHTAG = re.compile(r"#[\wÀ-ɏ゠-ヿ]+", re.UNICODE)
PADRAO_MENCAO = re.compile(r"@[\w.]+")
PADRAO_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]"
)
PADRAO_ACENTO_PT = re.compile(r"[ãõçáéêíóúâô]", re.IGNORECASE)
PADRAO_ARABE = re.compile(r"[؀-ۿ]")


def carregar_bruto(plataforma):
    return pd.read_json(DADOS_DIR / ARQUIVOS[plataforma], lines=True)


def _scores_keyframes(linha, categoria):
    valores = []
    for i in range(1, 6):
        quadro = linha[f"keyframe{i}"]
        if isinstance(quadro, list) and quadro:
            valores.append(quadro[0]["google"][categoria])
    return valores


def _features_visuais(df):
    saida = {}
    for categoria in CATEGORIAS_SAFESEARCH:
        listas = df.apply(lambda linha: _scores_keyframes(linha, categoria), axis=1)
        saida[f"kf_{categoria}_media"] = listas.apply(lambda v: np.mean(v) if v else np.nan)
        saida[f"kf_{categoria}_max"] = listas.apply(lambda v: np.max(v) if v else np.nan)
    return pd.DataFrame(saida, index=df.index)


def _proporcao(texto, condicao):
    letras = [c for c in texto if not c.isspace()]
    if not letras:
        return np.nan
    return sum(condicao(c) for c in letras) / len(letras)


def features_texto(legendas):
    legendas = legendas.fillna("").astype(str)
    hashtags = legendas.str.lower().apply(PADRAO_HASHTAG.findall)
    genericas = set(HASHTAGS_GENERICAS)
    texto_sem_tags = legendas.str.replace(PADRAO_HASHTAG, " ", regex=True).str.replace(
        PADRAO_MENCAO, " ", regex=True
    )
    return pd.DataFrame(
        {
            "tem_legenda": (legendas.str.strip() != "").astype(int),
            "comprimento_legenda": legendas.str.len(),
            "n_palavras": texto_sem_tags.str.split().str.len().fillna(0).astype(int),
            "n_hashtags": hashtags.str.len(),
            "n_hashtags_genericas": hashtags.apply(lambda t: sum(h in genericas for h in t)),
            "n_mencoes": legendas.apply(lambda s: len(PADRAO_MENCAO.findall(s))),
            "n_emojis": legendas.apply(lambda s: len(PADRAO_EMOJI.findall(s))),
            "tem_pergunta": legendas.str.contains(r"\?", regex=True).astype(int),
            "tem_exclamacao": legendas.str.contains("!", regex=False).astype(int),
            "prop_maiusculas": texto_sem_tags.apply(
                lambda s: _proporcao(s, str.isupper) if any(c.isalpha() for c in s) else np.nan
            ),
            "prop_ascii": legendas.apply(lambda s: _proporcao(s, str.isascii)),
            "tem_acento_pt": legendas.str.contains(PADRAO_ACENTO_PT).astype(int),
            "alfabeto_arabe": legendas.str.contains(PADRAO_ARABE).astype(int),
            "hashtags_texto": hashtags.apply(" ".join),
        },
        index=legendas.index,
    )


def montar_tiktok():
    bruto = carregar_bruto("tiktok")

    autor_ausente = (
        (bruto["author_followerCount"] == 0)
        & (bruto["author_heartCount"] == 0)
        & (bruto["author_videoCount"] == 0)
    )
    seguidores = bruto["author_followerCount"].astype(float).mask(autor_ausente)
    videos_autor = bruto["author_videoCount"].astype(float).mask(autor_ausente)
    curtidas_outros = (
        (bruto["author_heartCount"] - bruto["diggCount"]).clip(lower=0).astype(float).mask(autor_ausente)
    )

    criador = pd.DataFrame(
        {
            "seguidores": seguidores,
            "curtidas_autor_outros": curtidas_outros,
            "videos_autor": videos_autor,
            "curtidas_por_video_autor": curtidas_outros / videos_autor.clip(lower=1),
        }
    )

    engajamento = pd.DataFrame(
        {
            "curtidas": bruto["diggCount"],
            "comentarios": bruto["commentCount"],
            "compartilhamentos": bruto["shareCount"],
            "salvamentos": bruto["collectCount"],
        }
    )

    df = pd.concat(
        [
            bruto[["folder_id"]].rename(columns={"folder_id": "id"}),
            pd.to_datetime(bruto["day"], format="%d-%m-%Y").rename("dia_coleta"),
            criador,
            features_texto(bruto["desc"]),
            _features_visuais(bruto),
            engajamento,
            bruto["playCount"].rename("visualizacoes"),
            bruto["desc"].fillna("").rename("legenda"),
        ],
        axis=1,
    )
    return df


def rotular_viral(visualizacoes, limiar):
    return (visualizacoes >= limiar).astype(int)
