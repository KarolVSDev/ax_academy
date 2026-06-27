from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"

TRAIN_VALIDATION_FILE = DATA_DIR / "dataset_treino_validacao_padrao_realista.csv"
FINAL_TEST_FILE = DATA_DIR / "dataset_teste_final_padrao_realista.csv"

MODEL_FILE = MODEL_DIR / "modelo_vencedor.joblib"
METADATA_FILE = MODEL_DIR / "metadata_modelo.json"

TARGET = "situacao_final"

INPUT_FEATURES = [
    # Variáveis originais
    "frequencia_percent",
    "faltas",
    "atrasos",
    "horas_estudo_dia",
    "participacao_aulas",
    "atividades_entregues_percent",
    "apoio_familiar",
    "reprovacoes_anteriores",
    "nota_matematica",
    "nota_portugues",
    "nota_ciencias",
    "nota_humanas",
    "media",

    # Variáveis derivadas
    "media_notas",
    "menor_nota",
    "maior_nota",
    "qtd_notas_baixas",
    "frequencia_baixa",
    "muitas_faltas",
    "muitos_atrasos",
    "baixa_entrega_atividades",
    "baixa_participacao",
    "risco_academico",
    "diferenca_freq_atividades",
]

CLASS_LABELS = ["Aprovado", "Reprovado"]
POSITIVE_CLASS = "Reprovado"
