# 📚 Síntese dos estudos: base para a próxima versão do projeto

Leitura integral dos quatro artigos guardados em [`docs/estudos/`](.) (nov/2026). Para cada estudo, a síntese registra o que foi feito, o que foi encontrado, quanto vale a evidência e o que dá para aproveitar **com os dados e o escopo da disciplina** deste projeto. A seção 7 traz as propostas de melhoria, e a seção 8 compara este texto com o resumo inicial dos estudos.

---

## Visão geral

| | Ling et al. (2022) | Bravin et al. (2026) | Dubey et al. (2025) | Violot et al. (2024) |
|---|---|---|---|---|
| **Plataforma** | TikTok | TikTok | TikTok | YouTube |
| **Amostra** | 400 vídeos rotulados à mão | 86.244 vídeos de dança em 1.548 tendências, mais 3 experimentos | 1.515 vídeos sobre meio ambiente (inglês e espanhol) | 9,9 mi de Shorts e 6,9 mi de vídeos regulares, de 70 mil canais |
| **Variável estudada** | Viral × não viral (curtidas, extremos por hashtag) | Curtidas ÷ visualizações | Plays, curtidas, comentários e compartilhamentos (contagens) | Views, curtidas por view e comentários por view |
| **Método** | Codebook manual; Regressão Logística, SVM, árvore, Naive Bayes e RF (CV 10-fold) | Embeddings de vídeo (MUVID) e logit agrupado com efeitos fixos; experimentos online e de campo | Análise de conteúdo manual e regressão binomial negativa | Descritivo e longitudinal |
| **Achado central** | O nº de seguidores do criador é o maior preditor; plano fechado ou médio, texto na tela, câmera em 2ª pessoa e vídeo recente também se associam | Fugir do padrão da tendência aumenta o engajamento, sobretudo quando a tendência já é conhecida | Contexto global, relato pessoal, apelo emocional e posição explícita se associam a mais engajamento | Shorts têm mais views e mais curtidas por view, mas menos comentários por view; a vantagem varia por categoria |
| **Tipo de evidência** | Associação; amostra pequena e de extremos | Associação e **causa** (experimentos) | Associação; nicho ambiental | Descritiva |
| **Alcance ou engajamento?** | Curtidas absolutas (próximo de alcance) | Engajamento de quem já viu | Os dois, em contagens brutas | Os dois |

---

## 1. Ling et al. (2022): *Slapping Cats, Bopping Heads, and Oreo Shakes*

**Arquivo:** `01_ling_2022/ling_2022_slapping_cats_arxiv.pdf`. É o preprint do arXiv; a versão publicada saiu na WebSci '22, e o link da UCL aponta para o mesmo artigo.

**Desenho**

- Coletaram 103.587 vídeos das 63 hashtags da página *Descobrir* entre 11/03 e 11/04/2021. Mantiveram 28.342 vídeos de 20 hashtags que tinham vídeos virais e não virais.
- Em cada hashtag, os **10 vídeos com mais curtidas viraram "virais" e os 10 com menos, "não virais"**: 400 vídeos no total.
- Três pesquisadores criaram um codebook de 10 atributos, com kappa de Fleiss entre 0,63 e 0,89. Os atributos se dividem em três hipóteses:
  - **Conteúdo:** tema, enquadramento, ponto de vista, texto na tela, emoção e estilo.
  - **Recomendação:** tempo de vida do vídeo e hashtags.
  - **Criador:** se é verificado e se tem 10 mil seguidores ou mais.

**Resultados**

- A Regressão Logística chegou a AUC 0,93. Por grupo de atributos: só criador, 0,86; só conteúdo, 0,81; só recomendação, 0,71.
- Os cinco atributos mais importantes:

  | Atributo | Entre os virais | Entre os não virais |
  |---|---|---|
  | Criador com 10 mil seguidores ou mais | 90% | 24% |
  | Plano aberto | 1% | 17% |
  | Vídeo "no auge" do tempo de vida | 98% | 15% (85% eram "velhos demais") |
  | Texto na tela | 67% | 30% |
  | Câmera em 2ª pessoa (falando para quem assiste) | 70% | 60% |

- A emoção pesou pouco. Encher a legenda de hashtags não pareceu ajudar.

**Leitura crítica**

- O AUC alto vem do desenho. Separar os mais curtidos dos menos curtidos é bem mais fácil do que separar o top 20% de um feed. Os próprios autores dizem que os resultados "podem não ser tão claros" para vídeos do meio da distribuição.
- O "tempo de vida" mistura data e desenho amostral: os não virais eram vídeos antigos (o exemplo do artigo é de 2019) que ficaram no fim da busca. Além disso, a variável não existe no momento da postagem, então não serve para um modelo pré-publicação.
- Um único anotador rotulou 380 dos 400 vídeos. A generalização foi testada em só 70 vídeos.
- Curtidas absolutas misturam alcance e engajamento.

**Para o projeto:** confirma a ordem "criador mais forte que conteúdo" da nossa ablação (§9.2 do relatório) e a falta de efeito das hashtags genéricas (§13). Os atributos visuais (texto na tela, enquadramento, ponto de vista) não existem no dataset.

---

## 2. Bravin et al. (2026): *How Closely Should You Follow a Trend?*

**Arquivo:** `02_bravin_2026/bravin_2026_how_closely_follow_trend.pdf`. É o manuscrito aceito no *Journal of Marketing*, com licença CC BY-NC-ND 4.0. Os dados estão no Dataverse do JM (doi:10.7910/DVN/0M19OP).

**Desenho**

- **MUVID:** rede autossupervisionada que gera *embeddings* de 512 dimensões para imagem (R(2+1)D ResNet-18) e para áudio (ResNet-18 sobre espectrogramas). Foi treinada em 864.294 vídeos do TikTok.
- **Tendência:** agrupamento DBSCAN dos *embeddings* de áudio, isto é, vídeos que usam o mesmo trecho de música. Resultado: 86.244 vídeos de dança em 1.548 tendências.
- **Atipicidade:** distância cosseno média entre o vídeo e **todos os vídeos anteriores** da mesma tendência, no estilo de detecção de anomalias por k-NN.
- **Engajamento:** curtidas ÷ visualizações. O modelo é um logit agrupado (binomial) com efeitos fixos de criador, tendência e mês. Os controles são tempo na plataforma, qualidade visual, qualidade da dança, seguidores, selo de verificado e produtividade do criador.

**Resultados**

- **Sem modelo:** 111,9, 127,7 e 143,6 curtidas por mil views para atipicidade baixa, média e alta.
- **Com modelo:** cada desvio-padrão a mais de atipicidade multiplica a chance de uma view virar curtida por 1,025 (β = 0,025, p < 0,001, N = 74.227). O efeito é pequeno por vídeo, mas pesa em escala: os autores estimam cerca de 69 curtidas a mais por vídeo para um criador mediano.
- **Quando o efeito é maior:**
  - quando o público já viu muitos vídeos da tendência (interação β = 0,029);
  - quando a diferença aparece **no fim do vídeo**: nos últimos 5 s, β = 0,025; nos primeiros 5 s, sem efeito significativo.
- **Experimentos:**

  | Estudo | Desenho | Resultado |
  |---|---|---|
  | 2 | Online, N = 201 | 24% escolheram o vídeo atípico, contra 12,9% do típico |
  | 3 | Online, N = 494 | Efeito só com alta exposição à tendência: 34,2% × 17,6%. Com baixa exposição: 16,8% × 17,4% |
  | 4 | Campo, 38 criadores reais | Cerca de 10% a mais na chance de curtida (β = 0,093, p = 0,044) |

- Com efeito fixo de criador, a variação de seguidores ao longo do tempo não se associou à taxa de curtidas.

**Leitura crítica**

- É o único dos quatro estudos com evidência causal.
- O resultado vale para tendências de dança e mede engajamento de quem já viu o vídeo, não alcance.
- O estudo de campo tem poucos vídeos, com média de 783 views.

**Para o projeto**

- Separar **alcance** (nosso alvo atual) de **engajamento** (curtidas ÷ views).
- A atipicidade tem uma versão textual dentro do escopo da disciplina: a distância cosseno TF-IDF entre a legenda e as legendas anteriores com a mesma hashtag. É a mesma métrica de proximidade dos materiais de KNN. Ela mede só o texto, não a dança.
- A "exposição à tendência" pode ser aproximada pelo número de vídeos com a mesma hashtag nos dias anteriores do feed.

---

## 3. Dubey et al. (2025): *Talking Environment on TikTok*

**Arquivos:**
- `03_talking_environment_2025/talking_environment_tiktok_2025.pdf`: artigo em CC BY;
- `material_suplementar.pdf`: codebook e tabelas completas.

**Desenho**

- **Coleta:** 30 hashtags ambientais populares, com a lista em inglês traduzida para o espanhol, renderam 4.800 vídeos. Para cada hashtag, entraram os 30 vídeos do topo e os 10 mais recentes dos 5 maiores criadores. Depois de filtrar o que era de fato ambiental, ficaram 1.515 vídeos (699 em inglês e 816 em espanhol).
- **Codebook (baseado em Shifman, 2013):**
  - conteúdo: tema, ideologia e sentimento;
  - forma: áudio, vídeo e gênero;
  - postura: contexto, de quem é a experiência, tom e função comunicativa;
  - tipo de ator: pessoa, marca, ONG, governo etc.
- **Confiabilidade:** α de Krippendorff de 0,83 (inglês) e 0,71 (espanhol).
- **Modelo:** regressão binomial negativa pré-registrada para plays, curtidas, comentários e compartilhamentos.

**Resultados.** Os números são razões de taxa, exp(B), em relação à categoria de referência. Um traço (–) indica efeito não significativo ou não reportado no texto.

| Característica (referência) | Plays | Curtidas | Coment. | Compart. | % dos vídeos |
|---|---|---|---|---|---|
| Contexto global (contexto não claro) | 2,69 | 3,19 | 3,04 | 2,63 | 7,9% |
| Relato pessoal (não claro de quem é) | 1,93 | 1,80 | 2,74 | 2,04 | 35,9% |
| Função emotiva (informativa) | 3,56 | 4,64 | 4,55 | 3,65 | 5,4% |
| Ideologia explícita (implícita) | – | 1,54 (IC 0,94–2,52, n.s.) | 2,05 | 2,65 | 7,3% |
| Sentimento pró-ambiente (neutro) | **0,53** | **0,53** | – | – | 55,6% |
| Áudio diegético, som do próprio ambiente (áudio original) | 1,40 | 1,42 | – | – | 23,9% |
| Áudio de *performance*, dublagem ou lip-sync (áudio original) | – | – | – | 2,94 | 1,6% |
| Narração sobre imagens (apresentação) | – | – | 1,89 | – | 10,0% |
| Conta de governo (conta individual) | **0,09** | **0,07** | **0,12** | **0,15** | 0,7% |
| Organização de sustentabilidade (conta individual) | **0,19** | **0,16** | **0,17** | **0,12** | 6,3% |

**O idioma muda os efeitos:**
- humor funcionou em inglês, mas não em espanhol;
- chamada para ação teve cerca de 2,8× mais plays em espanhol;
- o tema alimentação foi bem pior em espanhol (0,03× plays);
- poluição teve mais comentários (5,6×) e compartilhamentos (15,2×) em espanhol.

**Leitura crítica**

- São associações num nicho específico (meio ambiente), com amostra puxada pelos vídeos mais populares de cada hashtag.
- Várias categorias têm poucos vídeos. A função emotiva, por exemplo, tem só 81, daí os intervalos de confiança largos.
- As contagens são brutas, e o texto não descreve controle por número de seguidores.

**Para o projeto**

- **Variáveis de mensagem tiradas da legenda.** São aproximações: a legenda não é o vídeo.
  - 1ª pessoa, como indício de relato pessoal;
  - palavras e emojis de emoção;
  - termos de alcance global ("mundo", "world", "planeta");
  - chamada para ação ("comente", "siga", "link na bio").
- **Checar se os efeitos mudam por idioma.** O dataset tem vídeos em português, inglês, espanhol e árabe.

---

## 4. Violot et al. (2024): *Shorts vs. Regular Videos on YouTube*

**Arquivo:** `04_violot_2024/violot_2024_shorts_vs_regular.pdf`. É a cópia do Desktop, publicada na WebSci '24 com licença CC BY 4.0.

**Desenho**

- 70.712 canais que postaram pelo menos um Short.
- Todos os vídeos desses canais entre jan/2021 e dez/2022: 9.883.770 Shorts e 6.862.321 vídeos regulares.
- Views, curtidas e comentários coletados uma única vez, em set/2023.
- Os Shorts foram identificados pelo redirecionamento da URL `/shorts/`.

**Resultados**

- **Concentração:** 1% dos Shorts ficam com 63% das views; nos vídeos regulares, 61%. A média engana, por isso o artigo usa a mediana.
- **Views:**
  - no geral, os Shorts tiveram 4× mais views médias que os vídeos regulares, chegando a 6× no fim de 2022;
  - dentro do mesmo canal, cerca de 110× mais: 80× no 1% de canais com mais inscritos e 111× nos demais.
- **Curtidas e comentários:**
  - curtidas por view começaram iguais; no fim de 2022, os Shorts tinham 1,4× mais;
  - comentários por view são maiores nos vídeos regulares, mas a diferença está diminuindo;
  - a correlação entre views e curtidas é de 0,85; entre views e comentários, de 0,27.
- **Categorias:**
  - os Shorts se concentram em Comédia (76,5% dos vídeos da categoria são Shorts) e em Pessoas e Blogs (74,3%); em Notícias e Política, só 14,4%;
  - a vantagem de views dos Shorts sobre os vídeos regulares fica abaixo da média em Música, Filmes e Animação, Games e Ciência e Tecnologia;
  - em Educação, a vantagem começa acima da média e cai;
  - em Sem Fins Lucrativos e Ativismo, sobe até 12,9×.
- **Duração:** pela mediana, vídeos regulares de 10 a 30 min (cerca de 5.800 views) superam os Shorts (1.986).

**Leitura crítica**

- O estudo é descritivo e não controla causas.
- A amostra foi semeada com palavras-chave de categorias do TikTok e inclui só canais que postaram ao menos um Short. As métricas são de um único instante.
- O próprio artigo se contradiz em dois pontos:
  - A conclusão diz que a vantagem dos Shorts é "menos marcante nas categorias de entretenimento", mas a Tabela 2 e a discussão mostram o contrário: entretenimento fica sempre acima da referência.
  - A discussão diz que Educação fica "consistentemente abaixo" da referência, mas na Tabela 2 ela está acima em 2021-S1 (5,28 contra 4,19).

**Para o projeto**

- O dataset tem 10.925 YouTube Shorts com categoria, duração, curtidas e comentários. Hoje eles são descartados porque não há tamanho do canal.
- Dá para replicar a comparação por categoria dentro dos Shorts brasileiros e medir a concentração de views.
- Não dá para comparar Shorts com vídeos regulares, porque o dataset só tem Shorts.

---

## 5. O que os quatro dizem juntos

1. **Alcance não é engajamento.**
   - Ling e o nosso projeto medem alcance (curtidas ou views absolutas).
   - Bravin e Violot medem taxas (curtidas ou comentários por view). Dubey usa as duas coisas.
   - As conclusões podem mudar de sinal entre um e outro. Numa checagem rápida no dataset completo (a refazer só no treino), seguidores × views dá correlação de Spearman 0,11, e seguidores × curtidas por view dá −0,13.
2. **O criador domina o alcance; o conteúdo ajusta na margem.**
   - Ling, em amostra de extremos: AUC 0,86 só com criador e 0,81 só com conteúdo.
   - Nossa ablação: 0,696 e 0,607.
   - Bravin: efeito de conteúdo pequeno, de 1,025 por desvio-padrão.
3. **O contexto modera tudo.** O efeito muda com a categoria (Violot), o idioma (Dubey) e a exposição à tendência (Bravin). Um efeito médio pode esconder efeitos opostos em grupos diferentes.
4. **Hashtags "para o algoritmo" não ajudam.** No Ling, o grupo de recomendação foi o mais fraco (AUC 0,71); na nossa EDA, também.
5. **Cauda pesada e seleção da amostra.**
   - No Violot, 1% dos Shorts concentra 63% das views. No Dubey, a média é 3,1 mi de plays e a mediana, 75,8 mil.
   - No nosso feed, 1% concentra 17,9% das views e a mediana é 1,8 mi. A amostra já é de vídeos recomendados.
   - Por isso separar o top 20% aqui é mais difícil do que separar extremos, e o nosso AUC de 0,72 não é comparável ao 0,93 do Ling.
6. **Só o Bravin testa causa.** O resto é associação, e a interface precisa continuar deixando isso claro.

---

## 6. Achado × o que dá para medir neste dataset

| Achado | Estudo | Dado disponível | Viável? | Como |
|---|---|---|---|---|
| Seguidores do criador (10 mil ou mais) | Ling | `author_followerCount` | ✅ já usado | Acrescentar `criador_popular_10k`, só para comparar com o artigo |
| Tempo de vida do vídeo | Ling | não há data de postagem | ❌ | E nem deveria entrar: no momento de postar, ele é zero |
| Texto na tela, enquadramento, ponto de vista | Ling | só escores SafeSearch dos quadros | ❌ | Exigiria os quadros-chave (sob solicitação aos autores do dataset) e visão computacional, fora da disciplina. Pode entrar na interface como checklist da literatura |
| Hashtags em alta | Ling | hashtags + `day` | ⚠️ aproximação | "Em alta" = frequente no feed nos dias anteriores |
| Atipicidade em relação à tendência | Bravin | legenda + `day` | ⚠️ só texto | Distância cosseno TF-IDF até as legendas anteriores da mesma hashtag |
| Exposição à tendência | Bravin | hashtags + `day` | ✅ | Nº de vídeos anteriores com a mesma hashtag |
| Taxa de engajamento como alvo | Bravin, Violot | `diggCount / playCount` | ✅ | Novo alvo: top 20% de curtidas por view |
| Relato pessoal, emoção, contexto global, chamada para ação | Dubey | legenda | ⚠️ aproximação | Léxico em português, inglês e espanhol, validado com amostra anotada à mão |
| Áudio dublado, narração sobre imagens | Dubey | não há áudio | ❌ | — |
| Tipo de ator (pessoa × organização) | Dubey | não há perfil | ❌ | — |
| Idioma modera os efeitos | Dubey | `prop_ascii`, `tem_acento_pt`, `alfabeto_arabe` | ✅ | Efeitos por grupo de idioma |
| Categoria modera; Shorts × vídeo regular | Violot | YouTube: `category_id`, `duration`, curtidas e comentários | ✅ categoria e duração / ❌ Shorts × regular | O dataset só tem Shorts |
| Concentração das views; média × mediana | Violot, Dubey | `playCount` | ✅ | Parte de 1% e de 10% no total de views |

---

## 7. Propostas para turbinar o projeto

Todas usam apenas técnicas dos materiais da disciplina: pipelines, os mesmos 14 modelos, TF-IDF e distância cosseno, K-Means e o protocolo 80/20 com CV.

| # | Proposta | Base | Como | Entrega | Esforço |
|---|---|---|---|---|---|
| **A1** | **Dois alvos: alcance × engajamento** | Bravin, Violot | Mesmo pipeline com o alvo `engajamento_alto` (top 20% de curtidas por view, limiar calculado só no treino). Comparar AUC e importâncias com o alvo de alcance | Seção nova no relatório, tabela e figura; aba na interface | Baixo–médio |
| **A2** | **Replicar o desenho do Ling** | Ling | Treinar e avaliar só com os extremos (top 10% × bottom 10% de views) e comparar o AUC com a amostra completa. Mostra quanto do 0,93 vem do desenho | Tabela e parágrafo na leitura crítica | Baixo |
| **A3** | **Tendência e atipicidade da legenda** | Bravin | Exposição à tendência (vídeos anteriores com a hashtag principal) e atipicidade (distância cosseno TF-IDF até as legendas anteriores da mesma hashtag), usando só dias anteriores, sem vazamento, e a interação entre as duas. Testar o ganho de AUC na CV | Novas variáveis em `features.py` e ablação | Médio |
| **B1** | **Variáveis de mensagem do Dubey** | Dubey | Léxicos de 1ª pessoa, emoção, alcance global e chamada para ação. Validar anotando à mão uns 100 vídeos e medindo a concordância (kappa), como fizeram os autores | Variáveis, tabela de validação e efeito médio | Médio |
| **B2** | **Módulo YouTube Shorts** | Violot | Views, curtidas por view e comentários por view por categoria e duração (com medianas); concentração das views; modelo sem dados do canal para mostrar o teto de previsão sem perfil | Script `src/youtube_shorts.py`, figuras e seção | Médio |
| **B3** | **Efeitos por idioma** | Dubey | Efeito médio das variáveis da legenda separado por idioma | Tabela | Baixo |
| **C1** | **Aba "O que a literatura diz" na interface** | os 4 | Checklist do que o modelo não mede (texto na tela, plano fechado ou médio, falar para a câmera, relato pessoal, versão própria da tendência, diferença no fim do vídeo), com estudo e força da evidência | `interface/index.html` | Baixo |
| **C2** | **Relatório** | os 4 | Reescrever o §11 com os quatro estudos, atualizar a referência do Ling para a versão publicada e acrescentar as limitações apontadas aqui | `reports/RELATORIO.md` | Baixo |

**Ordem sugerida:** A2 → A1 → A3 → C2 primeiro (é o núcleo de ML e o que mais pesa na avaliação), depois B1 e B2, e por fim B3 e C1. Antes de começar, vale conferir os requisitos do trabalho, para priorizar o que a avaliação pede.

---

## 8. Comparação com o resumo inicial

- **Ling:**
  - O link da UCL e o do arXiv são o mesmo artigo; a versão publicada é a da WebSci '22.
  - O "tempo de vida" é menos um fator de viralização e mais um efeito do desenho amostral. Também não está sob controle do criador nem existe antes de postar.
  - A regressão logística foi o melhor modelo (AUC 0,93), mas com amostra de extremos.
- **Bravin:** o resumo está correto. Vale acrescentar três pontos:
  - o efeito é pequeno (1,025 por desvio-padrão);
  - é mais forte no fim do vídeo e em tendências já conhecidas;
  - há evidência causal, com dois experimentos online e um de campo.
- **Dubey (Talking Environment):**
  - "Narração sobre imagens" aumentou **comentários**, não views.
  - O áudio de *performance* (dublagem ou lip-sync) aumentou **compartilhamentos**.
  - O tipo de áudio que aumentou plays e curtidas foi o **diegético** (som do próprio ambiente).
  - O sentimento pró-ambiente teve **metade** dos plays do sentimento neutro.
  - O link da Taylor & Francis pede acesso, mas o artigo é CC BY e está aberto no repositório da Universidade de Amsterdã.
- **Violot:** o resumo está correto no geral. A frase "vantagem menor em educação" vale de 2021-S2 em diante (ver a contradição na seção 4).

---

## Referências

1. Ling, C.; Blackburn, J.; De Cristofaro, E.; Stringhini, G. (2022). Slapping Cats, Bopping Heads, and Oreo Shakes: Understanding Indicators of Virality in TikTok Short Videos. *WebSci '22: 14th ACM Web Science Conference*. https://doi.org/10.1145/3501247.3531551 · preprint: arXiv:2111.02452 · UCL Discovery: https://discovery.ucl.ac.uk/id/eprint/10155763/
2. Bravin, M.; Clegg, M.; Hofstetter, R.; Pouly, M.; Berger, J. (2026). How Closely Should You Follow a Trend? Atypicality and Engagement on Social Media. *Journal of Marketing*. https://doi.org/10.1177/00222429261466668 · cópia aberta: https://doi.org/10.5281/zenodo.21905366 · dados: https://doi.org/10.7910/DVN/0M19OP
3. Dubey, S.; Meijers, M. H. C.; Smit, E. G.; Smit, E. S. (2025). Talking Environment on TikTok: Messages, Social Actors, and Engagement. *Environmental Communication*, 19(4), 723–748. https://doi.org/10.1080/17524032.2025.2453232 · codebook: https://osf.io/bewr4/
4. Violot, C.; Elmas, T.; Bilogrevic, I.; Humbert, M. (2024). Shorts vs. Regular Videos on YouTube: A Comparative Analysis of User Engagement and Content Creation Trends. *WebSci '24*, 213–223. https://doi.org/10.1145/3614419.3644023
