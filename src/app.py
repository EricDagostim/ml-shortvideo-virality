import argparse
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import FIGURAS_DIR, MODELOS_DIR, RAIZ, TABELAS_DIR
from src.recomendar import Avaliador

INTERFACE_DIR = RAIZ / "interface"

app = FastAPI(title="Viralidade TikTok — ML Clássico")
app.mount("/figuras", StaticFiles(directory=FIGURAS_DIR), name="figuras")

_avaliador = None


def avaliador():
    global _avaliador
    if _avaliador is None:
        if not (MODELOS_DIR / "modelo_escolhido.joblib").exists():
            raise HTTPException(503, "Modelo não treinado. Rode: python -m src.treinar_comparar")
        _avaliador = Avaliador()
    return _avaliador


class SafeSearch(BaseModel):
    adult: Optional[float] = Field(None, ge=0, le=4)
    medical: Optional[float] = Field(None, ge=0, le=4)
    spoofed: Optional[float] = Field(None, ge=0, le=4)
    violence: Optional[float] = Field(None, ge=0, le=4)
    racy: Optional[float] = Field(None, ge=0, le=4)


class Pedido(BaseModel):
    legenda: str = Field("", max_length=4000)
    seguidores: Optional[float] = Field(None, ge=0)
    videos_autor: Optional[float] = Field(None, ge=0)
    curtidas_autor: Optional[float] = Field(None, ge=0)
    safesearch: Optional[SafeSearch] = None


@app.get("/")
def pagina():
    return FileResponse(INTERFACE_DIR / "index.html")


@app.get("/api/modelo")
def modelo():
    return avaliador().resumo_modelo


@app.get("/api/comparacao")
def comparacao():
    caminho = TABELAS_DIR / "comparacao_modelos.csv"
    if not caminho.exists():
        raise HTTPException(404, "Comparação ainda não gerada")
    colunas = ["modelo", "auc_cv_media", "auc_cv_desvio", "auc_teste", "f1_teste", "precision_teste", "recall_teste", "tempo_fit_s", "escolhido"]
    tabela = pd.read_csv(caminho)[colunas]
    return tabela.to_dict(orient="records")


@app.get("/api/hashtags")
def hashtags():
    caminho = TABELAS_DIR / "efeito_hashtags.csv"
    if not caminho.exists():
        return []
    return pd.read_csv(caminho).to_dict(orient="records")


@app.post("/api/avaliar")
def avaliar(pedido: Pedido):
    safesearch = pedido.safesearch.model_dump() if pedido.safesearch else None
    return avaliador().avaliar(
        pedido.legenda,
        pedido.seguidores,
        pedido.videos_autor,
        pedido.curtidas_autor,
        safesearch,
    )


def main():
    parser = argparse.ArgumentParser(description="Interface local do assistente de viralidade.")
    parser.add_argument("--porta", type=int, default=8765)
    args = parser.parse_args()
    print(f"Interface em http://127.0.0.1:{args.porta}")
    uvicorn.run(app, host="127.0.0.1", port=args.porta)


if __name__ == "__main__":
    main()
