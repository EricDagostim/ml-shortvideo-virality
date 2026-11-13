"""Caminhos e constantes do modelo v3 (preditor de viralização antes de postar)."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
V3 = RAIZ / "data" / "v3"              # tudo aqui fica fora do git (conteúdo de terceiros, sem licença de redistribuição)
ABAS = V3 / "abas"                      # aba Shorts de cada canal (yt-dlp, lista plana)
CAPAS = V3 / "capas"                    # capas verticais (oar2.jpg) de cada Short
EMB = V3 / "embeddings"                 # vetores de título e de capa
MODELOS = RAIZ / "modelos" / "v3"
RELATORIO = RAIZ / "reports" / "v3"

VIDEOS_PROJ = RAIZ.parent / "videos"    # repositório do estúdio (Shorts do Gabriel, base faceless, momento)
SNAPUGC = VIDEOS_PROJ / "recursos" / "snapugc" / "base"
ESTUDIO = VIDEOS_PROJ / "estudio-shorts"
TOKENS = Path.home() / ".config" / "yt-publicar" / "tokens"

YTDLP = "/opt/homebrew/bin/yt-dlp"
SEMENTE = 42

# Rótulo: o Short "estourou" quando teve muitas vezes as views que o próprio canal costumava ter quando ele saiu.
# A régua do canal é a mediana dos Shorts anteriores (só o passado: é o que se sabe na hora de postar).
JANELA_REGUA = 30          # quantos Shorts anteriores formam a régua
MIN_ANTERIORES = 10        # sem isso, o canal ainda não tem régua e o Short fica fora do treino
IDADE_MIN_DIAS = 14        # Short mais novo que isso ainda está acumulando views
MULT_VIRAL = 5.0           # viral = views >= 5x a régua do canal...
VIEWS_MIN_VIRAL = 5000     # ...e pelo menos 5 mil views (passar de 120 para 1.800 não é viralizar)
MATURO_DIAS = 7            # features de resultado do canal só usam Shorts com >= 7 dias na hora de postar (views quase
                           # finais; é o que se sabe de verdade na véspera — evita aprender com o futuro dos vizinhos)
N_ANTERIORES_MIN_TREINO = 30   # viés de sobrevivência: a coleta achou canais que DERAM CERTO, então os primeiros Shorts
CANAL_IDADE_MIN_TREINO = 90    # deles estouram mais (13,5% x 9%). Esses Shorts iniciais ficam fora do treino.
REGUA_MIN = 300            # canais com régua abaixo disso ficam fora do treino e da avaliação (ruído de canal minúsculo)
