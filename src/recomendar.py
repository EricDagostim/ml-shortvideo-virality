import argparse
import json
from functools import cached_property

import joblib
import numpy as np
import pandas as pd

from src.config import CATEGORIAS_SAFESEARCH, HASHTAGS_GENERICAS, MODELOS_DIR, TABELAS_DIR
from src.divisao import carregar_divisao
from src.features import features_texto
from src.modelos import pontuacoes

DESCRICAO_FEATURES = {
    "comprimento_legenda": "caracteres na legenda",
    "n_palavras": "palavras (sem hashtags/menções)",
    "n_hashtags": "hashtags",
    "n_hashtags_genericas": "hashtags genéricas (#fyp, #viral...)",
    "n_mencoes": "menções (@)",
    "n_emojis": "emojis",
    "tem_pergunta": "tem pergunta",
    "tem_exclamacao": "tem exclamação",
    "prop_maiusculas": "proporção de letras maiúsculas",
    "tem_acento_pt": "tem acentuação do português (ã, ç, é...)",
}


def montar_linha(legenda, seguidores, videos_autor, curtidas_autor, visuais):
    texto = features_texto(pd.Series([legenda]))
    tem_autor = seguidores is not None and videos_autor is not None and curtidas_autor is not None
    criador = pd.DataFrame(
        {
            "seguidores": [seguidores],
            "curtidas_autor_outros": [curtidas_autor],
            "videos_autor": [videos_autor],
            "curtidas_por_video_autor": [curtidas_autor / max(videos_autor, 1) if tem_autor else np.nan],
        }
    ).astype(float)
    return pd.concat([criador, texto, pd.DataFrame([visuais])], axis=1)


def variantes(legenda, hashtags_sugeridas):
    tags = [t for t in legenda.split() if t.startswith("#")]
    texto = " ".join(t for t in legenda.split() if not t.startswith("#"))
    genericas = set(HASHTAGS_GENERICAS)
    cenarios = {"legenda original": legenda}
    sem_genericas = [t for t in tags if t.lower() not in genericas]
    if len(sem_genericas) != len(tags):
        cenarios["sem hashtags genéricas (#fyp, #viral...)"] = " ".join([texto] + sem_genericas).strip()
    if len(tags) > 3:
        cenarios["só as 3 primeiras hashtags"] = " ".join([texto] + tags[:3]).strip()
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


class Avaliador:
    def __init__(self):
        self.modelo = joblib.load(MODELOS_DIR / "modelo_escolhido.joblib")
        self.metadados = json.loads((MODELOS_DIR / "modelo_escolhido.json").read_text())
        self.colunas = self.metadados["colunas"]
        self.probabilistico = hasattr(self.modelo, "predict_proba")
        treino, _, self.limiar_viral = carregar_divisao()
        self.medianas_visuais = {
            f"kf_{c}_{a}": float(treino[f"kf_{c}_{a}"].median())
            for c in CATEGORIAS_SAFESEARCH
            for a in ("media", "max")
        }
        self.referencia = np.sort(pontuacoes(self.modelo, treino[self.colunas]))

    @cached_property
    def hashtags_sugeridas(self):
        caminho = TABELAS_DIR / "efeito_hashtags.csv"
        if not caminho.exists():
            return []
        efeitos = pd.read_csv(caminho)
        return efeitos[efeitos["delta_medio"] > 0]["hashtag"].tolist()

    @cached_property
    def resumo_modelo(self):
        caminho = TABELAS_DIR / "comparacao_modelos.csv"
        resumo = {
            "modelo": self.metadados["modelo"],
            "limiar_viral_visualizacoes": self.limiar_viral,
            "taxa_viral_base": self.metadados["taxa_viral_treino"],
            "probabilistico": self.probabilistico,
        }
        if caminho.exists():
            tabela = pd.read_csv(caminho).set_index("modelo")
            linha = tabela.loc[self.metadados["modelo"]]
            resumo.update(
                {
                    "auc_cv": float(linha["auc_cv_media"]),
                    "auc_teste": float(linha["auc_teste"]),
                    "f1_teste": float(linha["f1_teste"]),
                    "precision_teste": float(linha["precision_teste"]),
                    "recall_teste": float(linha["recall_teste"]),
                }
            )
        return resumo

    @cached_property
    def auc_por_subconjunto(self):
        caminho = TABELAS_DIR / "experimento_identidade_criador.csv"
        if not caminho.exists():
            return {}
        return pd.read_csv(caminho).set_index("subconjunto")["auc_teste"].to_dict()

    def confiabilidade(self, com_perfil):
        if com_perfil:
            auc = self.auc_por_subconjunto.get("criador nunca visto, com dados de perfil")
            return {
                "nivel": "moderada",
                "auc": auc,
                "mensagem": "Com os dados do perfil o modelo ordena bem melhor que o acaso, mas virais continuam difíceis de prever com certeza.",
            }
        auc = self.auc_por_subconjunto.get("sem dados do criador")
        return {
            "nivel": "baixa",
            "auc": auc,
            "mensagem": "Sem seguidores, curtidas e nº de vídeos do perfil o modelo fica próximo do acaso. Informe os três campos.",
        }

    def _visuais(self, safesearch):
        visuais = dict(self.medianas_visuais)
        for categoria, valor in (safesearch or {}).items():
            if valor is not None and categoria in CATEGORIAS_SAFESEARCH:
                visuais[f"kf_{categoria}_media"] = float(valor)
                visuais[f"kf_{categoria}_max"] = float(valor)
        return visuais

    def _escore(self, linha):
        return float(pontuacoes(self.modelo, linha[self.colunas])[0])

    def avaliar(self, legenda, seguidores=None, videos_autor=None, curtidas_autor=None, safesearch=None, top_hashtags=5):
        visuais = self._visuais(safesearch)
        linhas = []
        for nome, texto in variantes(legenda, self.hashtags_sugeridas[:top_hashtags]).items():
            linha = montar_linha(texto, seguidores, videos_autor, curtidas_autor, visuais)
            escore = self._escore(linha)
            linhas.append(
                {
                    "cenario": nome,
                    "legenda": texto,
                    "escore": escore,
                    "percentil": 100 * float(np.searchsorted(self.referencia, escore, side="right") / len(self.referencia)),
                    "previsto_viral": bool(escore >= self.metadados["limiar_decisao_f1"]),
                }
            )
        base = linhas[0]["escore"]
        for linha in linhas:
            linha["delta"] = linha["escore"] - base
        original = linhas[0]
        alternativas = sorted(linhas[1:], key=lambda l: l["delta"], reverse=True)

        caracteristicas = montar_linha(legenda, seguidores, videos_autor, curtidas_autor, visuais).iloc[0]
        return {
            "confiabilidade": self.confiabilidade(seguidores is not None and videos_autor is not None and curtidas_autor is not None),
            "resultado": original,
            "variacoes": alternativas,
            "caracteristicas": {
                rotulo: (None if pd.isna(caracteristicas[coluna]) else float(caracteristicas[coluna]))
                for coluna, rotulo in DESCRICAO_FEATURES.items()
            },
            "hashtags_detectadas": caracteristicas["hashtags_texto"].split(),
            "modelo": self.resumo_modelo,
        }


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

    avaliador = Avaliador()
    saida = avaliador.avaliar(
        args.legenda,
        args.seguidores,
        args.videos_autor,
        args.curtidas_autor,
        {c: getattr(args, c) for c in CATEGORIAS_SAFESEARCH},
        args.top_hashtags,
    )
    rotulo = "P(viral)" if avaliador.probabilistico else "escore"
    resumo = saida["modelo"]
    print(f"Modelo: {resumo['modelo']} | viral = top 20% (>= {resumo['limiar_viral_visualizacoes']:,.0f} visualizações)")
    print(f"Taxa base de viralização: {resumo['taxa_viral_base']:.1%}")
    conf = saida["confiabilidade"]
    print(f"Confiabilidade: {conf['nivel']} — {conf['mensagem']}\n")
    tabela = pd.DataFrame([saida["resultado"]] + saida["variacoes"]).drop(columns=["legenda"]).rename(columns={"escore": rotulo})
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "\nAtenção: as variações mostram associações aprendidas em dados observacionais; "
        "não garantem causa e efeito. Só adote hashtags coerentes com o conteúdo do vídeo e "
        "teste as melhores ideias na prática (ex.: A/B entre vídeos)."
    )


if __name__ == "__main__":
    main()
