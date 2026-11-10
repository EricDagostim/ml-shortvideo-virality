from sklearn.model_selection import train_test_split

from src.config import (
    COLUNA_ALVO,
    COLUNA_ALVO_CONTINUO,
    QUANTIL_VIRAL,
    RANDOM_STATE,
    TAMANHO_TESTE,
)
from src.features import montar_tiktok, rotular_viral


def carregar_divisao():
    df = montar_tiktok()
    treino, teste = train_test_split(df, test_size=TAMANHO_TESTE, random_state=RANDOM_STATE)
    limiar = treino[COLUNA_ALVO_CONTINUO].quantile(QUANTIL_VIRAL)
    treino = treino.assign(**{COLUNA_ALVO: rotular_viral(treino[COLUNA_ALVO_CONTINUO], limiar)})
    teste = teste.assign(**{COLUNA_ALVO: rotular_viral(teste[COLUNA_ALVO_CONTINUO], limiar)})
    return treino, teste, limiar
