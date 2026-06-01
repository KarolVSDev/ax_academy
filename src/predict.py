import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    CLASS_LABELS,
    FINAL_TEST_FILE,
    INPUT_FEATURES,
    METADATA_FILE,
    MODEL_FILE,
    POSITIVE_CLASS,
    TARGET,
)
from src.features import criar_variaveis_derivadas, validar_colunas


def carregar_modelo():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Modelo treinado não encontrado. Execute primeiro: python -m src.train"
        )

    return joblib.load(MODEL_FILE)


def carregar_metadata() -> dict:
    if not METADATA_FILE.exists():
        return {}

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def carregar_teste_final() -> pd.DataFrame:
    if not FINAL_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de teste final não encontrado: {FINAL_TEST_FILE}"
        )

    df = pd.read_csv(FINAL_TEST_FILE)
    df = criar_variaveis_derivadas(df)
    validar_colunas(df, INPUT_FEATURES + [TARGET], "teste final")

    return df


def gerar_amostra(
    df: pd.DataFrame,
    percentual: int,
    random_state: int = 42,
) -> pd.DataFrame:
    percentual = max(1, min(100, int(percentual)))
    n_amostra = max(1, round(len(df) * percentual / 100))

    return df.sample(n=n_amostra, random_state=random_state).copy()


def executar_inferencia(
    percentual: int,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    modelo = carregar_modelo()
    df_teste = carregar_teste_final()
    amostra = gerar_amostra(df_teste, percentual, random_state=random_state)

    X = amostra[INPUT_FEATURES]
    y_real = amostra[TARGET]
    y_pred = modelo.predict(X)

    resultado = amostra.copy()
    resultado["valor_real"] = y_real
    resultado["valor_previsto"] = y_pred
    resultado["acertou"] = resultado["valor_real"] == resultado["valor_previsto"]

    if hasattr(modelo, "predict_proba"):
        probas = modelo.predict_proba(X)
        classes = modelo.classes_

        for i, classe in enumerate(classes):
            resultado[f"prob_{classe}"] = probas[:, i]

    metricas = {
        "total_amostra": int(len(resultado)),
        "percentual_amostrado": percentual,
        "acuracia": accuracy_score(y_real, y_pred),
        "precisao_reprovado": precision_score(
            y_real, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "recall_reprovado": recall_score(
            y_real, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "f1_reprovado": f1_score(
            y_real, y_pred, pos_label=POSITIVE_CLASS, zero_division=0
        ),
        "matriz_confusao": confusion_matrix(
            y_real,
            y_pred,
            labels=CLASS_LABELS,
        ).tolist(),
        "relatorio_classificacao": classification_report(
            y_real,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }

    return resultado, metricas
