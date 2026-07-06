import json
import warnings

import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    FINAL_TEST_FILE,
    INPUT_FEATURES,
    METADATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    POSITIVE_CLASS,
    TARGET,
    TRAIN_VALIDATION_FILE,
)
from src.features import criar_variaveis_derivadas, validar_colunas

warnings.filterwarnings("ignore")


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de treino/validação não encontrado: {TRAIN_VALIDATION_FILE}"
        )

    if not FINAL_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de teste final não encontrado: {FINAL_TEST_FILE}"
        )

    df_tv = pd.read_csv(TRAIN_VALIDATION_FILE)
    df_teste = pd.read_csv(FINAL_TEST_FILE)

    df_tv = criar_variaveis_derivadas(df_tv)
    df_teste = criar_variaveis_derivadas(df_teste)

    validar_colunas(df_tv, INPUT_FEATURES + [TARGET], "treino/validação")
    validar_colunas(df_teste, INPUT_FEATURES + [TARGET], "teste final")

    return df_tv, df_teste


def avaliar_modelo(nome_modelo: str, modelo, X_val, y_val) -> dict:
    y_pred = modelo.predict(X_val)

    return {
        "modelo": nome_modelo,
        "acuracia": accuracy_score(y_val, y_pred),
        "precisao_reprovado": precision_score(
            y_val, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "recall_reprovado": recall_score(
            y_val, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "f1_reprovado": f1_score(
            y_val, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
    }


def criar_modelos() -> dict:
    modelos = {
        "Regressão Logística": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Árvore de Decisão": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    DecisionTreeClassifier(
                        max_depth=6,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "modelo",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "SVM Linear": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    SVC(
                        kernel="linear",
                        C=1,
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "SVM RBF C=1": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    SVC(
                        kernel="rbf",
                        C=1,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Rede Neural MLP 32-16": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.001,
                        max_iter=700,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    modelos["Voting Classifier"] = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "modelo",
                VotingClassifier(
                    estimators=[
                        (
                            "logistica",
                            LogisticRegression(
                                max_iter=1000,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                        (
                            "arvore",
                            DecisionTreeClassifier(
                                max_depth=6,
                                min_samples_leaf=5,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                        (
                            "random_forest",
                            RandomForestClassifier(
                                n_estimators=250,
                                max_depth=10,
                                min_samples_leaf=3,
                                class_weight="balanced",
                                random_state=42,
                            ),
                        ),
                    ],
                    voting="soft",
                ),
            ),
        ]
    )

    modelos_smote = {
        "Regressão Logística + SMOTE": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest + SMOTE": RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
        ),
        "Extra Trees + SMOTE": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    for nome, modelo in modelos_smote.items():
        modelos[nome] = ImbPipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("modelo", modelo),
            ]
        )

    return modelos


def treinar_e_salvar(percentual_treino: float = 0.70) -> dict:
    df_tv, df_teste = carregar_dados()

    if "id_aluno" in df_tv.columns and "id_aluno" in df_teste.columns:
        ids_tv = set(df_tv["id_aluno"])
        ids_teste = set(df_teste["id_aluno"])
        intersecao = ids_tv.intersection(ids_teste)
        if intersecao:
            raise ValueError(
                "Há IDs repetidos entre treino/validação e teste final. "
                "O teste precisa permanecer inédito."
            )

    X = df_tv[INPUT_FEATURES]
    y = df_tv[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=percentual_treino,
        random_state=42,
        stratify=y,
    )

    modelos = criar_modelos()
    resultados = []
    modelos_treinados = {}

    for nome, modelo in modelos.items():
        print(f"Treinando: {nome}")
        modelo.fit(X_train, y_train)

        resultado = avaliar_modelo(nome, modelo, X_val, y_val)
        resultados.append(resultado)
        modelos_treinados[nome] = modelo

    df_resultados = (
        pd.DataFrame(resultados)
        .sort_values(by="f1_reprovado", ascending=False)
        .reset_index(drop=True)
    )

    melhor_nome = df_resultados.loc[0, "modelo"]
    melhor_modelo = modelos_treinados[melhor_nome]

    X_test = df_teste[INPUT_FEATURES]
    y_test = df_teste[TARGET]
    y_pred_test = melhor_modelo.predict(X_test)

    resultado_teste = {
        "modelo": melhor_nome,
        "acuracia_teste": accuracy_score(y_test, y_pred_test),
        "precisao_reprovado": precision_score(
            y_test, y_pred_test, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "recall_reprovado": recall_score(
            y_test, y_pred_test, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "f1_reprovado": f1_score(
            y_test, y_pred_test, pos_label=POSITIVE_CLASS, zero_division=0
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(melhor_modelo, MODEL_FILE)

    metadata = {
        "modelo_vencedor": melhor_nome,
        "percentual_treino": percentual_treino,
        "total_treino_validacao": int(len(df_tv)),
        "total_teste_final": int(len(df_teste)),
        "resultados_validacao": df_resultados.to_dict(orient="records"),
        "resultado_teste_final": resultado_teste,
        "classification_report_teste": classification_report(
            y_test,
            y_pred_test,
            zero_division=0,
            output_dict=True,
        ),
        "variaveis_entrada": INPUT_FEATURES,
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    df_resultados.to_csv(
        "resultados_validacao.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([resultado_teste]).to_csv(
        "resultado_teste_final.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nModelo vencedor:", melhor_nome)
    print("Modelo salvo em:", MODEL_FILE)
    print("Metadata salva em:", METADATA_FILE)
    print("\nResultado no teste final:")
    print(pd.DataFrame([resultado_teste]))

    return metadata


if __name__ == "__main__":
    treinar_e_salvar()
