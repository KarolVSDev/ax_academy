import json
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from src.config import (
    CLASS_LABELS,
    FINAL_TEST_FILE,
    METADATA_FILE,
    MODEL_FILE,
    TRAIN_VALIDATION_FILE,
)
from src.predict import executar_inferencia
from src.train import treinar_e_salvar


st.set_page_config(
    page_title="MVP Risco Pedagógico",
    page_icon="",
    layout="wide",
)


def formatar_percentual(valor: float) -> str:
    return f"{valor * 100:.2f}%"


def arquivo_existe(path: Path) -> bool:
    return Path(path).exists()


def carregar_metadata() -> dict:
    if not METADATA_FILE.exists():
        return {}

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def renderizar_matriz_confusao(matriz):
    matriz = np.asarray(matriz)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=CLASS_LABELS,
    )
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Matriz de Confusão — Amostra Selecionada")
    st.pyplot(fig)

st.title("Classificação de Risco Pedagógico")
st.caption(
    "MVP de Machine Learning para prever a situação final do aluno: "
    "Aprovado ou Reprovado."
)

with st.sidebar:
    st.header("Configuração da demonstração")

    percentual = st.slider(
        "Percentual do teste final para amostragem",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
    )

    random_state = st.number_input(
        "Semente aleatória",
        min_value=1,
        max_value=9999,
        value=42,
        step=1,
    )

    st.divider()

    st.subheader("Status dos arquivos")
    st.write(
        "Treino/validação:",
        " encontrado" if arquivo_existe(TRAIN_VALIDATION_FILE) else " ausente",
    )
    st.write(
        "Teste final:",
        " encontrado" if arquivo_existe(FINAL_TEST_FILE) else " ausente",
    )
    st.write(
        "Modelo treinado:",
        " encontrado" if arquivo_existe(MODEL_FILE) else " ausente",
    )

    treinar_agora = st.button("Treinar/atualizar modelo")


if treinar_agora:
    with st.spinner("Treinando modelos e selecionando o melhor pela validação..."):
        try:
            metadata_treino = treinar_e_salvar()
            st.success(
                f"Modelo treinado com sucesso: {metadata_treino['modelo_vencedor']}"
            )
        except Exception as erro:
            st.error(f"Não foi possível treinar o modelo: {erro}")


metadata = carregar_metadata()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Problema de negócio")
    st.write(
        "Identificar alunos com risco de reprovação para apoiar decisões "
        "pedagógicas preventivas."
    )

with col2:
    st.markdown("### Técnica aplicada")
    st.write(
        "Pipeline de classificação supervisionada com imputação, padronização "
        "quando necessária, comparação de modelos e escolha pelo F1-score da "
        "classe Reprovado."
    )

with col3:
    st.markdown("### Uso do teste final")
    st.write(
        "O conjunto de teste fica separado do treinamento. A banca escolhe um "
        "percentual, e o sistema sorteia essa fração para inferência em tempo real."
    )

st.divider()

if not arquivo_existe(TRAIN_VALIDATION_FILE) or not arquivo_existe(FINAL_TEST_FILE):
    st.warning(
        "Coloque os dois arquivos CSV dentro de `data/processed/` para executar o MVP."
    )
    st.code(
        "data/processed/dataset_treino_validacao_padrao_realista.csv\n"
        "data/processed/dataset_teste_final_padrao_realista.csv"
    )
    st.stop()

if not arquivo_existe(MODEL_FILE):
    st.info(
        "O modelo ainda não foi treinado. Clique em **Treinar/atualizar modelo** "
        "na barra lateral ou execute `python -m src.train` no terminal."
    )
    st.stop()

try:
    df_predicoes, metricas = executar_inferencia(
        percentual=percentual,
        random_state=int(random_state),
    )
except Exception as erro:
    st.error(f"Erro ao executar inferência: {erro}")
    st.stop()

modelo_vencedor = metadata.get("modelo_vencedor", "Modelo treinado")
resultado_teste = metadata.get("resultado_teste_final", {})

st.subheader("Resultado da inferência em tempo real")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Modelo", modelo_vencedor)
m2.metric("Amostra", metricas["total_amostra"])
m3.metric("Acurácia", formatar_percentual(metricas["acuracia"]))
m4.metric("Recall Reprovado", formatar_percentual(metricas["recall_reprovado"]))
m5.metric("F1 Reprovado", formatar_percentual(metricas["f1_reprovado"]))

with st.expander("Métricas gerais do modelo no teste final completo"):
    if resultado_teste:
        st.dataframe(pd.DataFrame([resultado_teste]), width="stretch")
    else:
        st.write("Metadata do teste final ainda não encontrada.")

tab1, tab2, tab3 = st.tabs(
    [" Métricas da amostra", " Predições", " Download"]
)

with tab1:
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("#### Matriz de confusão")
        renderizar_matriz_confusao(metricas["matriz_confusao"])

    with c2:
        st.markdown("#### Distribuição das predições")
        distribuicao = (
            df_predicoes["valor_previsto"]
            .value_counts()
            .rename_axis("classe_prevista")
            .reset_index(name="quantidade")
        )
        st.bar_chart(distribuicao, x="classe_prevista", y="quantidade")

    st.markdown("#### Métricas detalhadas")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "percentual_amostrado": f"{metricas['percentual_amostrado']}%",
                    "total_amostra": metricas["total_amostra"],
                    "acuracia": metricas["acuracia"],
                    "precisao_reprovado": metricas["precisao_reprovado"],
                    "recall_reprovado": metricas["recall_reprovado"],
                    "f1_reprovado": metricas["f1_reprovado"],
                }
            ]
        ),
        width="stretch",
    )

with tab2:
    st.markdown("#### Classes reais, classes preditas e probabilidade")
    colunas_prioritarias = [
        "id_aluno",
        "valor_real",
        "valor_previsto",
        "acertou",
        "frequencia_percent",
        "faltas",
        "atrasos",
        "media",
        "risco_academico",
    ]

    colunas_prob = [col for col in df_predicoes.columns if col.startswith("prob_")]
    colunas_exibir = [
        col for col in colunas_prioritarias + colunas_prob if col in df_predicoes.columns
    ]

    st.dataframe(
        df_predicoes[colunas_exibir],
        width="stretch",
        hide_index=True,
    )

with tab3:
    csv = df_predicoes.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Baixar predições da amostra em CSV",
        data=csv,
        file_name="predicoes_amostra.csv",
        mime="text/csv",
    )

st.divider()
st.markdown(
    "Desenvolvido por Ana Karoline como parte do projeto de Risco Pedagógico. "
    "Código disponível no [GitHub])."
)
