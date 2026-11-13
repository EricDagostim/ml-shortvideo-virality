"""Exporta os Shorts do canal Simulando o Mundo para `data/canal/simulando_o_mundo.csv`.

Lê os dados que a pipeline de análise do estúdio já coletou (YouTube Data API + Analytics API)
e grava uma linha por vídeo público, com as mesmas informações que o dataset do Zenodo tem
(duração, categoria, idioma, descrição) e as métricas reais para comparar com o modelo.

Uso:
    python -m src.canal                       # procura ../videos/estudio-shorts
    python -m src.canal --estudio CAMINHO     # outro local

O CSV contém dados do YouTube Analytics do canal e está no .gitignore. Só tire de lá
se quiser publicar esses números no repositório.
"""

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from src.config import RAIZ

CANAL = "simulando"
ESTUDIO_PADRAO = RAIZ.parent / "videos" / "estudio-shorts"
SAIDA = RAIZ / "data" / "canal" / "simulando_o_mundo.csv"


def _ultimo_bruto(dados: Path, video_id: str) -> dict:
    """Resposta mais recente da Data API gravada para o vídeo (contagens públicas + ficha)."""
    arquivos = sorted(glob.glob(str(dados / "coletas" / video_id / "*_data_api*.json")))
    for caminho in reversed(arquivos):
        bruto = json.loads(Path(caminho).read_text()).get("bruto")
        if bruto:
            return bruto
    return {}


def _valor(metricas: dict, chave: str):
    item = metricas.get(chave)
    return item.get("valor") if isinstance(item, dict) else None


def _views_ate_dia(dias: list, n: int):
    """Views acumuladas até o n-ésimo dia do Analytics (dia 1 = dia da publicação, fuso do YouTube)."""
    if not dias or len(dias) < n:
        return None
    return dias[n - 1]["acumulado"].get("views")


def exportar(estudio: Path = ESTUDIO_PADRAO) -> pd.DataFrame:
    dados = estudio / "analise" / "dados"
    registro = json.loads((dados / "videos.json").read_text())["videos"]
    consolidado = json.loads((dados / "consolidado.json").read_text())["videos"]

    linhas = []
    for video_id, ficha in registro.items():
        if ficha.get("canal") != CANAL or ficha.get("privacidade") != "public":
            continue
        bruto = _ultimo_bruto(dados, video_id)
        snippet = bruto.get("snippet", {})
        estatisticas = bruto.get("statistics", {})
        cons = consolidado.get(video_id, {})
        metricas = cons.get("metricas", {})
        carac = cons.get("caracteristicas") or {}
        fala = carac.get("fala") or {}
        cenas = carac.get("cenas") or {}
        entrega = ficha.get("entrega") or ""

        linhas.append({
            "video_id": video_id,
            "producao": entrega.rsplit("_", 1)[-1] if entrega else None,
            "titulo": ficha.get("titulo"),
            "publicado_em": ficha.get("publicado_em"),
            # mesmas informações que o dataset do Zenodo tem
            "duracao_s": ficha.get("duracao_s"),
            "category_id": int(ficha["categoria"]) if ficha.get("categoria") else None,
            "defaultLanguage": snippet.get("defaultLanguage") or "",
            "description": ficha.get("descricao") or "",
            # resultado real
            "views": int(estatisticas["viewCount"]) if "viewCount" in estatisticas else None,
            "likes": int(estatisticas["likeCount"]) if "likeCount" in estatisticas else None,
            "comentarios": int(estatisticas["commentCount"]) if "commentCount" in estatisticas else None,
            "views_dia1": _views_ate_dia(cons.get("dias"), 1),
            "views_ate_dia2": _views_ate_dia(cons.get("dias"), 2),
            "retencao_media_pct": _valor(metricas, "average_view_percentage"),
            "tempo_medio_visto_s": _valor(metricas, "average_view_duration_s"),
            "inscritos_ganhos": _valor(metricas, "subscribers_gained"),
            "compartilhamentos": _valor(metricas, "shares"),
            # como o vídeo foi feito (características extraídas pela pipeline)
            "voz": (carac.get("voz") or {}).get("provedor"),
            "palavras_por_minuto": fala.get("wpm_medio"),
            "planos": cenas.get("planos"),
            "pergunta_no_gancho": fala.get("pergunta_no_hook"),
        })

    tabela = pd.DataFrame(linhas).sort_values("publicado_em").reset_index(drop=True)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(SAIDA, index=False)
    return tabela


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--estudio", type=Path, default=ESTUDIO_PADRAO)
    args = parser.parse_args()
    tabela = exportar(args.estudio)
    print(f"{len(tabela)} vídeos → {SAIDA.relative_to(RAIZ)}")
    print(tabela[["producao", "duracao_s", "views", "views_ate_dia2", "retencao_media_pct", "voz"]].to_string())
