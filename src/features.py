import pandas as pd


def criar_variaveis_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Cria variáveis derivadas usadas pelo modelo.

    A função não utiliza a variável alvo, evitando vazamento de informação.
    """
    df = df.copy()

    colunas_notas = [
        "nota_matematica",
        "nota_portugues",
        "nota_ciencias",
        "nota_humanas",
    ]

    colunas_ausentes = [col for col in colunas_notas if col not in df.columns]
    if colunas_ausentes:
        raise ValueError(f"Colunas de notas ausentes: {colunas_ausentes}")

    df["media_notas"] = df[colunas_notas].mean(axis=1)
    df["menor_nota"] = df[colunas_notas].min(axis=1)
    df["maior_nota"] = df[colunas_notas].max(axis=1)

    df["qtd_notas_baixas"] = (df[colunas_notas] < 6).sum(axis=1)

    df["frequencia_baixa"] = (df["frequencia_percent"] < 75).astype(int)
    df["muitas_faltas"] = (df["faltas"] >= 15).astype(int)
    df["muitos_atrasos"] = (df["atrasos"] >= 10).astype(int)

    df["baixa_entrega_atividades"] = (
        df["atividades_entregues_percent"] < 70
    ).astype(int)

    df["baixa_participacao"] = (
        df["participacao_aulas"] < 70
    ).astype(int)

    df["risco_academico"] = (
        df["frequencia_baixa"]
        + df["muitas_faltas"]
        + df["muitos_atrasos"]
        + df["baixa_entrega_atividades"]
        + df["baixa_participacao"]
        + df["qtd_notas_baixas"]
        + df["reprovacoes_anteriores"]
    )

    df["diferenca_freq_atividades"] = (
        df["frequencia_percent"] - df["atividades_entregues_percent"]
    )

    return df


def validar_colunas(df: pd.DataFrame, colunas: list[str], nome_base: str) -> None:
    """Valida se as colunas necessárias existem no DataFrame."""
    faltantes = [col for col in colunas if col not in df.columns]
    if faltantes:
        raise ValueError(
            f"A base {nome_base} não possui as colunas obrigatórias: {faltantes}"
        )
