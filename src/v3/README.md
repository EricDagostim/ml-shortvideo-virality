# v3 — Preditor de viralização ANTES de postar (Shorts animados e faceless)

Pergunta: **"este Short vai estourar no meu canal?"**, respondida antes de publicar, com a probabilidade calibrada e
um veredito (VIRAL / PROMISSOR / NEUTRO / NÃO VIRAL) cujos limiares foram escolhidos para ter poucos falsos positivos.

## Arquitetura

```
                ┌──────────────────────────── BASE (coleta.py, base.py) ───────────────────────────┐
 buscas no      │ 523 canais de Shorts do nicho (explicações animadas, "e se", ciência, curiosidades;│
 YouTube  ───►  │ PT/EN/ES/DE/FR) → aba Shorts completa (yt-dlp) → Data API (data, duração, língua, │
 + sementes     │ tags, views) → 123 mil Shorts; 87 mil no treino (2022+, régua, ≥ 14 dias, canal   │
 da base v2     │ com ≥ 30 Shorts anteriores e ≥ 90 dias)                                           │
                └──────────────────────────────────────────────────────────────────────────────────┘
                         │                      │                         │
                ┌────────▼───────┐   ┌──────────▼──────────┐   ┌──────────▼───────────┐
                │ título         │   │ capa (hqdefault)    │   │ conteúdo do vídeo    │
                │ mpnet multilín-│   │ CLIP ViT-B/32       │   │ 3 quadros a 25/50/75%│
                │ gue (768)      │   │ + nota "animado"    │   │ (hq1-3.jpg) → CLIP,  │
                │ (embeddings.py)│   │ (embeddings.py)     │   │ cor, texto, diversid.│
                └────────┬───────┘   └──────────┬──────────┘   │ (quadros.py)         │
                         │                      │              └──────────┬───────────┘
          ┌──────────────▼──────────────────────▼─────────────────────────▼─────────────┐
          │ FEATURES (features.py) — só o que existe antes de postar                     │
          │  canal     régua, dispersão, taxa de estouros, tendência, ritmo (Shorts ≥ 7 d)│
          │  título    padrões (hipótese, pergunta, número...), PCA, PCA relativo ao canal │
          │  meta      duração, hora, dia, descrição, tags, língua                         │
          │  capa      PCA do CLIP, animado                                                │
          │  quadros   PCA do CLIP, animado, diversidade visual, cor, bordas (texto)       │
          │  novidade  parecença com o que o canal já postou + resultado dos parecidos     │
          │  demanda   kNN: títulos parecidos de OUTROS canais, publicados ≥ 14 dias antes │
          │  cabeças   ridge no vetor inteiro de título / capa / quadros (empilhamento)    │
          └──────────────────────────────────┬──────────────────────────────────────────┘
                                             │
                         ┌───────────────────▼──────────────────┐
                         │ par LightGBM (pesos por canal):       │
                         │  classificador "estoura?" +           │
                         │  regressão do múltiplo da régua       │
                         │  → média dos postos = nota            │
                         │ + calibração isotônica (fora da dobra)│
                         │ + limiares por precisão (cross-fit)   │
                         └───────────────────┬──────────────────┘
                                             │
                ┌────────────────────────────▼─────────────────────────────┐
                │ prever.py: pasta da entrega (post.json + MP4 + capa)       │
                │  → histórico do canal pela API → mesmas features → veredito│
                │  → o que puxou a nota → temas parecidos em outros canais   │
                └───────────────────────────────────────────────────────────┘
```

**Rótulo.** `régua` = mediana das views dos 30 Shorts anteriores do mesmo canal (só o passado). O Short **estourou**
quando `views ≥ 5 × régua` **e** `views ≥ 5.000`. Isso tira o tamanho do canal da conta: no Simulando (régua ~1.360),
estourar é passar de ~6.800 views. Taxa base no treino: 9,0%.

**Três cuidados contra números bonitos e falsos** (cada um derrubou a métrica e foi mantido):
- *Futuro dos vizinhos.* As features de resultado do canal (dispersão, taxa de estouros, tendência, régua) só usam
  Shorts com ≥ 7 dias na hora de postar. Com 2 dias, o treino via as views finais de Shorts que, na véspera, ainda
  estavam subindo — o modelo aprendia "a onda que vem aí". AUC caiu de 0,737 para 0,710 com o corte.
- *Viés de sobrevivência.* A coleta achou canais que deram certo, então os primeiros Shorts deles estouram mais
  (13,5% nos canais com < 30 dias contra 9%). Shorts com < 30 anteriores ou de canal com < 90 dias ficam fora do treino,
  e a idade do canal e o nº de Shorts anteriores não entram como features (num canal novo, só extrapolariam).
- *Canal novo.* Quando o canal ainda não tem 10 Shorts com ≥ 7 dias, o contexto usa os que têm ≥ 2 dias.

**Por que não usar o resto dos datasets do projeto como linhas de treino.**
- Zenodo YouTube Shorts BR (10.925): está "desidratado", sem id, título nem canal; só dá para usar como referência de
  categoria e duração.
- TikTok BR (15.193): outra plataforma e outro algoritmo; o rótulo de alcance não se transfere para Shorts.
- Base v2 (384 Shorts faceless): os 54 canais dela entraram como **sementes** da coleta e estão todos na v3.
- Momento (curvas das primeiras horas de ~500 Shorts): é pós-publicação; fica como etapa 2 opcional (abaixo).

## Validação (treinar.py)

- Treinar só com canais animados foi pior do que com a base inteira, mesmo medindo nos animados (AUC 0,720 x 0,737
  no experimento): mais dados venceram a especialização.
- Árvores pequenas (15 folhas, ≥ 400 Shorts por folha) + regressão do múltiplo da régua, combinadas pela média dos
  postos, deram mais precisão no topo que um classificador de 63 folhas (top 1%: 37% x 32%; `data/v3/exp/variantes.log`).
- 5 dobras **por canal**: nenhum canal aparece no treino e no teste (o Simulando não está na base).
- Demanda e cabeças calculadas dentro de cada dobra, só com canais do treino; para as linhas de treino, as cabeças
  vêm de validação aninhada por canal.
- Vizinhos da demanda só de vídeos publicados ≥ 14 dias antes; resultados do próprio canal só de Shorts com ≥ 2 dias.
- Limiar de cada dobra escolhido nas outras dobras: a precisão reportada é fora da amostra.
- AUC **dentro do canal** (separar os Shorts de um mesmo canal) e teste temporal (treina antes de jun/2026, testa
  depois em canais nunca vistos).

Resultados (detalhes em `reports/v3/RELATORIO_V3.md`; números em `reports/v3/validacao.json`):

| | |
|---|---|
| AUC / average precision (base 9,0%) | 0,698 / 0,195 (teste temporal: 0,696 / 0,193) |
| VIRAL (limiar escolhido fora da dobra) | 0,8% dos Shorts, **38,6% estouram** (4,3× a média), recall 3,6% |
| PROMISSOR | 17% dos Shorts, 20% estouram |
| NÃO VIRAL | 36% dos Shorts, 3,6% estouram |
| Calibração (decil de cima) | previsto 23,5%, real 23,4% |
| Simulando (5 Shorts auditados) | 5 NEUTRO, nenhum estourou, nenhum falso positivo |

## Como rodar

```bash
.venv/bin/python -m src.v3.coleta tudo          # canais, abas, API, capas (~2 h)
.venv/bin/python -m src.v3.base                 # base_v3.parquet
.venv/bin/python -m src.v3.embeddings           # vetores de título e capa
.venv/bin/python -m src.v3.quadros --paralelo 28   # quadros do vídeo (~3 h)
caffeinate -i .venv/bin/python -m src.v3.treinar   # validação + modelo final (~15 min; caffeinate: o Mac em repouso
                                                    # congela o treino)
.venv/bin/python -m src.v3.relatorio            # figuras e tabelas

# antes de postar:
.venv/bin/python -m src.v3.prever --entrega ../videos/entregas/simulando/<pasta>
.venv/bin/python -m src.v3.prever --canal simulando --titulo "E se ..." --titulo "Outra versão"   # só ideias (supõe o visual médio do canal)
.venv/bin/python -m src.v3.prever --canal simulando --auditar                                  # Shorts já postados
```

## Etapa 2 (opcional, depois de postar)

O preditor de momento (`videos/recursos/snapugc/mac/momento_prever.py`) lê as views da 1ª, 3ª e 6ª hora e acerta
estourou/não estourou em 76%, 81% e 85% dos Shorts (24 h: 95%). Não substitui o veredito antes de postar: confirma.

## Limites

- O algoritmo do YouTube testa cada Short com um público pequeno e decide pelo comportamento desse público; parte do
  resultado é sorte e não está no vídeo. O teto de um preditor pré-publicação é baixo, e o veredito VIRAL é raro de
  propósito (ver a tabela de decisão no relatório).
- Os dados de vídeo usados no treino são 3 quadros, não o vídeo inteiro (baixar o vídeo em massa esbarra no bloqueio
  de robô do YouTube, que não foi contornado). Áudio, narração e ritmo de cortes ficam de fora.
- `data/v3/` não vai para o git (conteúdo de terceiros).
