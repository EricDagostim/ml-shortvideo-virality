from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DADOS_DIR = RAIZ / "data" / "raw"
MODELOS_DIR = RAIZ / "modelos"
FIGURAS_DIR = RAIZ / "reports" / "figuras"
TABELAS_DIR = RAIZ / "reports" / "tabelas"

ZENODO_RECORD = "15446544"
ARQUIVOS = {
    "tiktok": "tiktok_metadata_dehydrated.jsonl",
    "youtube": "youtube_metadata_dehydrated.jsonl",
}

RANDOM_STATE = 42
TAMANHO_TESTE = 0.20
QUANTIL_VIRAL = 0.80
N_FOLDS = 5
N_BOOTSTRAP = 1000

CATEGORIAS_SAFESEARCH = ["adult", "medical", "spoofed", "violence", "racy"]

HASHTAGS_GENERICAS = [
    "#fyp", "#foryou", "#foryoupage", "#fy", "#fypシ", "#viral", "#parati",
    "#trending", "#viralvideo", "#viral_video", "#viraltiktok", "#tiktok",
    "#explore", "#fypage", "#fyy", "#trend", "#xyzbca", "#capcut",
]

COLUNAS_CRIADOR = [
    "seguidores",
    "curtidas_autor_outros",
    "videos_autor",
    "curtidas_por_video_autor",
]

COLUNAS_TEXTO_CONTAGEM = [
    "comprimento_legenda",
    "n_palavras",
    "n_hashtags",
    "n_hashtags_genericas",
    "n_mencoes",
    "n_emojis",
]

COLUNAS_TEXTO_OUTRAS = [
    "tem_legenda",
    "tem_pergunta",
    "tem_exclamacao",
    "prop_maiusculas",
    "prop_ascii",
    "tem_acento_pt",
    "alfabeto_arabe",
]

COLUNAS_VISUAIS = [f"kf_{c}_{a}" for c in CATEGORIAS_SAFESEARCH for a in ("media", "max")]

COLUNA_HASHTAGS = "hashtags_texto"

COLUNAS_ENGAJAMENTO = ["curtidas", "comentarios", "compartilhamentos", "salvamentos"]

COLUNA_ALVO_CONTINUO = "visualizacoes"
COLUNA_ALVO = "viral"
