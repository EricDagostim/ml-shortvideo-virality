# 📚 Documentos de referência

## Materiais da disciplina (`materiais_disciplina/`)

Material do Prof. Dr. Rodrigo Ramos Silva (Machine Learning Clássico, 2026). Ele define o **escopo permitido** de algoritmos e técnicas deste trabalho. Nenhum método fora desse material foi usado na modelagem.

| Arquivo | Conteúdo | Onde foi aplicado no trabalho |
|---|---|---|
| `Aula_-_KNN` · `Guia_Pratico_Aula_KNN_Python_Estruturas_Dados_FINAL` | KNN, distâncias, escolha de K, padronização, leakage específico | Modelo `KNN` com `StandardScaler` no pipeline; busca conjunta de K, métrica (p=1/2) e pesos |
| `KNN_Exemplos_Avancados_COM_CODIGO` · `KNN_Exemplos_Avancados_Metricas_Aplicacoes_Analise` | Métricas de proximidade (L1, L2, Minkowski, cosseno/TF-IDF, Jaccard...) | Escolha de `p` por validação; hashtags representadas como vetor binário (`TfidfVectorizer` binário) |
| `Aula_6_SVM_com_regressao_SVR` · `Guia_Pratico_SVM_Explicado_do_Zero_Versao_Simples` · `Guia_SVC_Etapas_Equacoes_Anotadas` | SVC/SVR, margem, C, kernel RBF, gamma, normalização | Modelo `SVM (SVC RBF)` com busca de C e gamma via `GridSearchCV` + `StratifiedKFold` |
| `IA-aula-arvore-decisao-26` · `Guia_Pratico_Arvores_Decisao_Classificacao_Regressao_2026` | Entropia, Gini, CART, poda (`max_depth`, `min_samples_leaf`, `ccp_alpha`) | Modelo `Árvore de Decisão` com busca de critério, profundidade e poda |
| `Ensembles_Aula_completo` · `Guia_Ensemble_Learning_AMPLIADO_Explicado_Exemplos_Python` | Voting, Bagging, RF, Extra Trees, AdaBoost, GB, HistGB, XGBoost, Stacking (OOF), protocolo de avaliação, permutation importance | 9 ensembles comparados; protocolo 80/20 + CV estratificada (§9.6); `permutation_importance` × `feature_importances_` |
| `k-means-com-introducao-clustering` · `Guia_Pratico_Clustering_KMeans_Python_Comentado` · `caso-clustering-clientes-do-inicio-ao-fim` | K-Means, cotovelo, silhueta, ARI, estabilidade, PCA, perfis e cautela causal | `src/clustering.py`: perfis de conteúdo, escolha de K, estabilidade por ARI, PCA |

O gabarito da prova N1, que também faz parte do material, **não foi publicado**, porque o repositório é público. Os conceitos dele (ordem do pipeline, *fit* só no treino, métricas por custo do erro, não ajustar nada no teste) foram seguidos no protocolo.

## Artigos citados

- **Dataset:** Lacerda, T. et al. *Exploring Brazilian TikTok and YouTube Shorts: A Public Dataset for Video Characterization*. BraSNAM 2025. DOI [10.5753/brasnam.2025.9244](https://doi.org/10.5753/brasnam.2025.9244) · dados: [10.5281/zenodo.15446544](https://doi.org/10.5281/zenodo.15446544)
- As demais referências estão na seção de referências de [`../reports/RELATORIO.md`](../reports/RELATORIO.md).
