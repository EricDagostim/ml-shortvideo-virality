# 🎬 O que faz um vídeo curto viralizar? — Machine Learning Clássico no TikTok

Trabalho de Machine Learning que analisa **15.193 vídeos reais do TikTok** (feed "Para Você", Brasil, nov/2024–dez/2024) para (1) entender quais padrões estão associados à viralização e (2) construir um algoritmo que ajuda a avaliar e ajustar um vídeo **antes de publicar**.

Foram treinados e comparados **14 algoritmos** vistos na disciplina (de uma Regressão Logística até XGBoost, Voting e Stacking), com o mesmo pré-processamento, a mesma validação cruzada e o mesmo conjunto de teste reservado. O relatório completo, com a justificativa do modelo escolhido, está em **[`reports/RELATORIO.md`](reports/RELATORIO.md)**.

---

## 📦 Dados

| | |
|---|---|
| **Fonte** | Lacerda, T.; Locatelli, M. S.; Costa, I.; Carneiro, L.; Almeida, V.; Meira Jr., W. *Exploring Brazilian TikTok and YouTube Shorts: A Public Dataset for Video Characterization*. BraSNAM 2025, SBC. DOI [10.5753/brasnam.2025.9244](https://doi.org/10.5753/brasnam.2025.9244) |
| **Download** | Zenodo, DOI [10.5281/zenodo.15446544](https://doi.org/10.5281/zenodo.15446544) — licença CC-BY-4.0, uso acadêmico |
| **Plataformas** | TikTok (15.193 vídeos) e YouTube Shorts (10.925 vídeos) |
| **Plataforma analisada** | **TikTok** — é a mais completa (ver abaixo) |

**Por que só TikTok?** A proposta era separar a análise por plataforma usando só a que tivesse dados mais completos. No YouTube Shorts, `description` está vazia em 66% dos vídeos, `defaultLanguage` em 78%, `caption` é `false` em 99,5%, `dislike_count` é sempre 0 e **não existe o tamanho do canal**. O TikTok tem mais vídeos, métricas do criador (seguidores, curtidas e nº de vídeos) e legenda presente em 93% dos casos. Tabela completa: [`reports/tabelas/completude_plataformas.csv`](reports/tabelas/completude_plataformas.csv).

**Por que não o dataset do Kaggle `what-makes-a-yt-video-go-viral-12000-vids`?** Ele foi avaliado e descartado: é **sintético** (o próprio autor desaconselha usá-lo para análise real), contém **apenas YouTube** e os vídeos **não são curtos** (mediana de 614 s; só 23 dos 12.000 têm até 60 s).

## 🎯 Definição do problema

- **Alvo:** `viral = 1` se o vídeo está entre os **20% mais vistos** (≥ 6,9 milhões de visualizações). O limiar é calculado **só no treino** e aplicado ao teste.
- **Tipo:** classificação binária supervisionada (classe positiva = viral, 20%).
- **Entradas permitidas:** somente informações disponíveis **antes da publicação**:
  - **Criador:** seguidores, curtidas de outros vídeos, nº de vídeos, curtidas por vídeo. Quando os três campos vêm zerados juntos (20,8% dos vídeos), são tratados como **ausentes**.
  - **Legenda:** tamanho, palavras, hashtags, hashtags genéricas (#fyp, #viral...), menções, emojis, pergunta, exclamação, caixa alta, idioma/alfabeto e as 30 hashtags mais frequentes.
  - **Visual:** escores do Google SafeSearch (adult, medical, spoofed, violence, racy) nos 5 quadros-chave.
- **Entradas proibidas (vazamento):** curtidas, comentários, compartilhamentos e salvamentos do próprio vídeo. Elas só existem depois da publicação e crescem junto com as visualizações. O experimento de vazamento mostra o efeito de incluí-las.

## 🧪 Protocolo experimental

Segue o protocolo dos materiais da disciplina (Guia de Ensembles §9, gabarito da N1, Q4/Q5/Q14):

1. Divisão **80% treino / 20% teste** (`random_state=42`); o teste só é usado na avaliação final.
2. EDA, limiar do alvo, imputação, escala e vocabulário de hashtags são ajustados **só no treino**, dentro de um `Pipeline` + `ColumnTransformer`.
3. Tuning de hiperparâmetros com `GridSearchCV`/`RandomizedSearchCV` e `StratifiedKFold(5)` no treino.
4. **Métrica de seleção:** ROC-AUC médio na CV, que mede a capacidade de **ordenar** ideias de vídeo por chance de viralizar, independentemente do limiar. Como métricas de decisão, F1, Precision e Recall são reportados com o limiar escolhido por F1 nas previsões *out-of-fold* do treino.
5. **Regra de escolha (definida antes de ver os resultados):** maior AUC na CV; entre os modelos a até **1 erro-padrão** do melhor, escolhe-se o mais simples.
6. Confirmação no teste com **IC 95% por bootstrap pareado** (1.000 reamostragens) da diferença de AUC entre o escolhido e cada concorrente.

## 📁 Estrutura

```
src/
├── config.py                     # caminhos, constantes e grupos de colunas
├── baixar_dados.py               # baixa o dataset do Zenodo e valida o MD5
├── features.py                   # engenharia de features (criador, legenda, visual)
├── divisao.py                    # divisão treino/teste e limiar viral calculado no treino
├── eda.py                        # análise exploratória (somente treino) + completude das plataformas
├── modelos.py                    # pré-processamento + 14 modelos e espaços de busca
├── treinar_comparar.py           # tuning, CV, teste, bootstrap e escolha do modelo
├── experimentos_complementares.py# vazamento, ablação criador×conteúdo e validação temporal
├── interpretar.py                # permutation importance, importância por impureza, efeito médio, hashtags
├── clustering.py                 # K-Means de perfis de conteúdo (cotovelo, silhueta, ARI, PCA)
├── treinar_escolhido.py          # recria só o modelo vencedor a partir dos hiperparâmetros salvos (≈1 min)
├── recomendar.py                 # núcleo do assistente + CLI: avalia uma legenda e sugere variações
├── app.py                        # interface local (FastAPI) que serve interface/index.html
└── v3/                           # preditor antes de postar, YouTube Shorts (ver src/v3/README.md)
interface/
└── index.html                    # página do assistente (HTML + JS puro, sem build)
docs/
├── README.md                     # índice dos materiais e onde cada um foi aplicado
└── materiais_disciplina/         # slides e guias do professor (escopo dos algoritmos)
reports/
├── RELATORIO.md                  # relatório final
├── v3/                           # relatório, figuras e tabelas da v3
├── figuras/                      # gráficos gerados
└── tabelas/                      # resultados em CSV
modelos/                          # modelos treinados (.joblib, não versionados)
```

## ▶️ Como rodar

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m src.baixar_dados
.venv/bin/python -m src.eda
.venv/bin/python -m src.treinar_comparar
.venv/bin/python -m src.experimentos_complementares
.venv/bin/python -m src.interpretar
.venv/bin/python -m src.clustering
.venv/bin/python -m src.app
```

A comparação completa leva cerca de 30–40 min em 16 núcleos.

### 🖥️ Interface local (recomendado)

```bash
.venv/bin/python -m src.app
```

Na primeira execução, se o dataset ou o modelo não existirem, o app baixa os dados do Zenodo e treina o modelo escolhido com os hiperparâmetros vencedores salvos em `reports/tabelas/comparacao_modelos.csv` (≈ 1 minuto, via `src/treinar_escolhido.py`). Não é preciso rodar a comparação inteira para usar o assistente.

Abra **http://127.0.0.1:8765** (ou escolha outra porta com `--porta 9000`). A página tem três abas:

- **Avaliar vídeo:** cole a legenda, informe seu perfil (seguidores, nº de vídeos e curtidas totais) e veja a probabilidade de o vídeo ficar entre os 20% mais vistos, o percentil, a **confiabilidade** da previsão e as variações da legenda que o modelo estima que aumentam ou diminuem a chance. O botão **usar** aplica a variação e reavalia.
- **Comparação dos modelos:** a tabela dos 14 algoritmos e os gráficos da comparação.
- **Hashtags:** o efeito médio estimado de adicionar cada uma das hashtags mais frequentes.

Também dá para abrir já avaliando, via URL: `http://127.0.0.1:8765/?legenda=minha%20legenda%20%23fyp&seguidores=50000&videos=120&curtidas=900000`

![Interface](reports/figuras/15_interface.png)

> Sem os três campos do perfil a confiabilidade é **baixa** (AUC ≈ 0,56): o conteúdo da legenda sozinho carrega pouco sinal. Com o perfil, AUC ≈ 0,74 em criadores nunca vistos no treino.

### Usando o algoritmo pela linha de comando

```bash
.venv/bin/python -m src.recomendar \
  --legenda "Testei a receita mais viral do momento 😱 #receita #cozinha #fyp #viral" \
  --seguidores 50000 --videos-autor 120 --curtidas-autor 900000
```

A saída traz a probabilidade de viralizar, o percentil em relação aos vídeos do treino e o efeito estimado de variações da legenda (remover hashtags genéricas, adicionar hashtags associadas a alcance, transformar o texto em pergunta etc.).

## 🚀 v3 — "este Short vai estourar?" antes de postar (YouTube Shorts)

O TikTok (acima) não responde à pergunta de um canal pequeno de Shorts animados. A v2
([`notebooks/projeto_final_v2_conteudo.ipynb`](notebooks/projeto_final_v2_conteudo.ipynb)) usou 384 Shorts de 54 canais
e chegou a AUC ≈ 0,58. A v3 junta uma base bem maior e muda o alvo para a régua do próprio canal:

- **Base:** 123.178 Shorts de 523 canais do nicho (explicação animada, "e se", ciência, curiosidade; PT/EN/ES/DE/FR),
  coletados pela aba Shorts de cada canal + YouTube Data API. Os 54 canais da v2 entraram como sementes.
- **Alvo:** o Short **estourou** quando fez ≥ 5× a mediana dos 30 Shorts anteriores do canal **e** ≥ 5 mil views
  (9,0% da base). Assim o tamanho do canal sai da conta.
- **Entradas (só o que existe antes de postar):** contexto do canal, título (vetor multilíngue), capa (CLIP), 3 quadros
  do vídeo (CLIP, cor, texto na tela), parecença com o que o canal já postou e com o que deu certo em outros canais.
- **Modelo:** par LightGBM (classificador + regressão do múltiplo da régua), calibração isotônica e limiares de decisão
  escolhidos fora da amostra.
- **Validação:** 5 dobras por canal (o modelo é sempre testado em canais que nunca viu) e teste temporal.

| veredito | % dos Shorts | estouram de fato |
|---|---|---|
| VIRAL | 0,8% | **38,6%** (4,3× a média) |
| PROMISSOR | 17% | 20% |
| NÃO VIRAL | 36% | 3,6% |

AUC 0,698 (teste temporal 0,696); calibração no decil de cima: 23,5% previsto, 23,4% real. No canal *Simulando o
Mundo*, os 5 Shorts auditáveis saíram NEUTRO e nenhum estourou (nenhum falso positivo).

```bash
.venv/bin/python -m src.v3.prever --entrega ../videos/entregas/simulando/<pasta>          # vídeo pronto
.venv/bin/python -m src.v3.prever --canal simulando --titulo "E se ..." --titulo "..."      # só ideias
```

Relatório: [`reports/v3/RELATORIO_V3.md`](reports/v3/RELATORIO_V3.md) · arquitetura e como rodar:
[`src/v3/README.md`](src/v3/README.md) · notebook:
[`notebooks/projeto_final_v3_pre_postagem.ipynb`](notebooks/projeto_final_v3_pre_postagem.ipynb). Os dados coletados
(`data/v3/`) e o modelo (`modelos/v3/`) ficam fora do git.

## 🛠️ Stack

Python 3 · pandas · NumPy · SciPy · scikit-learn · XGBoost · Matplotlib · Seaborn · joblib · FastAPI + Uvicorn (interface)
· v3: LightGBM · PyTorch · sentence-transformers · OpenCLIP · OpenCV · YouTube Data API
