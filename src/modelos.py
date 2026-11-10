import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import (
    COLUNA_HASHTAGS,
    COLUNAS_CRIADOR,
    COLUNAS_ENGAJAMENTO,
    COLUNAS_TEXTO_CONTAGEM,
    COLUNAS_TEXTO_OUTRAS,
    COLUNAS_VISUAIS,
    RANDOM_STATE,
)

N_HASHTAGS_VOCABULARIO = 30


def colunas_entrada(criador=True, conteudo=True, engajamento=False):
    colunas = []
    if criador:
        colunas += COLUNAS_CRIADOR
    if conteudo:
        colunas += COLUNAS_TEXTO_CONTAGEM + COLUNAS_TEXTO_OUTRAS + COLUNAS_VISUAIS + [COLUNA_HASHTAGS]
    if engajamento:
        colunas += COLUNAS_ENGAJAMENTO
    return colunas


def criar_preprocessador(criador=True, conteudo=True, engajamento=False):
    colunas_log = []
    if criador:
        colunas_log += COLUNAS_CRIADOR
    if conteudo:
        colunas_log += COLUNAS_TEXTO_CONTAGEM
    if engajamento:
        colunas_log += COLUNAS_ENGAJAMENTO

    transformadores = []
    if colunas_log:
        transformadores.append(
            (
                "contagens_log",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("imputador", SimpleImputer(strategy="median", add_indicator=True)),
                        ("escala", StandardScaler()),
                    ]
                ),
                colunas_log,
            )
        )
    if conteudo:
        transformadores.append(
            (
                "numericas",
                Pipeline(
                    [
                        ("imputador", SimpleImputer(strategy="median")),
                        ("escala", StandardScaler()),
                    ]
                ),
                COLUNAS_TEXTO_OUTRAS + COLUNAS_VISUAIS,
            )
        )
        transformadores.append(
            (
                "hashtags",
                TfidfVectorizer(
                    token_pattern=r"#[^\s#]+",
                    lowercase=True,
                    binary=True,
                    use_idf=False,
                    norm=None,
                    max_features=N_HASHTAGS_VOCABULARIO,
                ),
                COLUNA_HASHTAGS,
            )
        )
    return ColumnTransformer(transformadores, sparse_threshold=0.0)


def _pipeline(estimador, **kwargs_prep):
    return Pipeline([("prep", criar_preprocessador(**kwargs_prep)), ("modelo", estimador)])


def catalogo_modelos():
    rs = RANDOM_STATE
    return {
        "Dummy (baseline)": {
            "estimador": DummyClassifier(strategy="prior"),
            "espaco": None,
            "complexidade": 0,
        },
        "Regressão Logística": {
            "estimador": LogisticRegression(max_iter=5000, class_weight="balanced"),
            "espaco": {"modelo__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 10.0]},
            "complexidade": 1,
        },
        "Árvore de Decisão": {
            "estimador": DecisionTreeClassifier(class_weight="balanced", random_state=rs),
            "espaco": {
                "modelo__max_depth": [3, 4, 5, 6, 8, 10, 14, None],
                "modelo__min_samples_leaf": [20, 50, 100, 200, 400, 800],
                "modelo__ccp_alpha": [0.0, 1e-4, 3e-4, 1e-3],
                "modelo__criterion": ["gini", "entropy"],
            },
            "n_iter": 25,
            "complexidade": 2,
        },
        "KNN": {
            "estimador": KNeighborsClassifier(),
            "espaco": {
                "modelo__n_neighbors": [51, 101, 201, 301, 451, 601, 801],
                "modelo__weights": ["uniform", "distance"],
                "modelo__p": [1, 2],
            },
            "complexidade": 3,
        },
        "SVM (SVC RBF)": {
            "estimador": SVC(kernel="rbf", class_weight="balanced", cache_size=1000),
            "espaco": {
                "modelo__C": [0.3, 1.0, 3.0],
                "modelo__gamma": [0.005, 0.02, 0.05, 0.1],
            },
            "complexidade": 4,
        },
        "AdaBoost": {
            "estimador": AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=rs), random_state=rs
            ),
            "espaco": {
                "modelo__estimator__max_depth": [1, 2, 3, 4],
                "modelo__n_estimators": [200, 400, 800, 1600],
                "modelo__learning_rate": [0.03, 0.1, 0.3, 1.0],
            },
            "n_iter": 15,
            "complexidade": 5,
        },
        "Bagging (árvores)": {
            "estimador": BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=rs),
                n_estimators=300,
                bootstrap=True,
                random_state=rs,
                n_jobs=1,
            ),
            "espaco": {
                "modelo__max_samples": [0.3, 0.5, 0.8, 1.0],
                "modelo__max_features": [0.5, 0.8, 1.0],
                "modelo__estimator__min_samples_leaf": [1, 5, 10, 20, 50, 100],
            },
            "n_iter": 15,
            "complexidade": 6,
        },
        "Random Forest": {
            "estimador": RandomForestClassifier(n_estimators=400, random_state=rs, n_jobs=1),
            "espaco": {
                "modelo__max_depth": [None, 8, 12, 20],
                "modelo__min_samples_leaf": [1, 3, 5, 10, 20, 40, 80],
                "modelo__max_features": ["sqrt", 0.2, 0.35, 0.5, 0.8],
                "modelo__class_weight": [None, "balanced_subsample"],
            },
            "n_iter": 25,
            "complexidade": 6,
        },
        "Extra Trees": {
            "estimador": ExtraTreesClassifier(n_estimators=400, random_state=rs, n_jobs=1),
            "espaco": {
                "modelo__max_depth": [None, 8, 12, 20],
                "modelo__min_samples_leaf": [1, 3, 5, 10, 20, 40, 80],
                "modelo__max_features": ["sqrt", 0.35, 0.5, 0.8, 1.0],
                "modelo__class_weight": [None, "balanced_subsample"],
            },
            "n_iter": 25,
            "complexidade": 6,
        },
        "Gradient Boosting": {
            "estimador": GradientBoostingClassifier(random_state=rs),
            "espaco": {
                "modelo__n_estimators": [150, 300, 600, 1000],
                "modelo__learning_rate": [0.01, 0.02, 0.05, 0.1],
                "modelo__max_depth": [2, 3, 4, 5, 6],
                "modelo__subsample": [0.7, 0.85, 1.0],
                "modelo__min_samples_leaf": [1, 20, 50],
            },
            "n_iter": 20,
            "complexidade": 7,
        },
        "HistGradientBoosting": {
            "estimador": HistGradientBoostingClassifier(random_state=rs),
            "espaco": {
                "modelo__learning_rate": [0.01, 0.02, 0.05, 0.1],
                "modelo__max_iter": [150, 300, 600, 1000],
                "modelo__max_leaf_nodes": [8, 15, 31, 63],
                "modelo__min_samples_leaf": [10, 20, 50, 100],
                "modelo__l2_regularization": [0.0, 0.1, 1.0, 5.0],
            },
            "n_iter": 25,
            "complexidade": 7,
        },
        "XGBoost": {
            "estimador": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=rs,
                n_jobs=1,
                tree_method="hist",
            ),
            "espaco": {
                "modelo__n_estimators": [150, 300, 500, 800, 1200],
                "modelo__max_depth": [2, 3, 4, 5, 6],
                "modelo__learning_rate": [0.01, 0.02, 0.04, 0.08],
                "modelo__subsample": [0.7, 0.85, 1.0],
                "modelo__colsample_bytree": [0.5, 0.7, 0.85, 1.0],
                "modelo__reg_lambda": [0.5, 1.0, 2.0, 5.0],
                "modelo__min_child_weight": [1, 5, 10, 20, 40],
            },
            "n_iter": 30,
            "complexidade": 7,
        },
    }


def montar_ensembles(melhores):
    def modelo_de(nome):
        return clone(melhores[nome].named_steps["modelo"])

    componentes = [
        ("logistica", modelo_de("Regressão Logística")),
        ("knn", modelo_de("KNN")),
        ("random_forest", modelo_de("Random Forest")),
        ("xgboost", modelo_de("XGBoost")),
    ]
    return {
        "Voting (soft)": {
            "estimador": VotingClassifier(estimators=componentes, voting="soft", n_jobs=1),
            "espaco": None,
            "complexidade": 9,
        },
        "Stacking": {
            "estimador": StackingClassifier(
                estimators=componentes,
                final_estimator=LogisticRegression(max_iter=5000),
                cv=5,
                stack_method="predict_proba",
                n_jobs=1,
            ),
            "espaco": None,
            "complexidade": 10,
        },
    }


def pipeline_para(estimador, **kwargs_prep):
    return _pipeline(clone(estimador), **kwargs_prep)


def pontuacoes(modelo, X):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    return modelo.decision_function(X)
