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
    st.markdown("### Simulação individual da predição")
    st.write(
        "Selecione um aluno da amostra para visualizar quais dados entram no modelo "
        "e qual é a saída gerada pela predição."
    )

    df_visual = df_predicoes.copy()

    # Garante que existe uma coluna de identificação para exibição
    if "id_aluno" in df_visual.columns:
        opcoes_alunos = df_visual["id_aluno"].tolist()

        aluno_selecionado = st.selectbox(
            "Selecione o aluno para análise",
            opcoes_alunos,
        )

        aluno = df_visual[df_visual["id_aluno"] == aluno_selecionado].iloc[0]
    else:
        opcoes_alunos = df_visual.index.tolist()

        aluno_selecionado = st.selectbox(
            "Selecione o registro para análise",
            opcoes_alunos,
        )

        aluno = df_visual.loc[aluno_selecionado]

    st.divider()

    col_entrada, col_saida = st.columns([1.2, 1])

    with col_entrada:
        st.markdown("#### Entrada do modelo")
        st.caption("Dados do aluno usados para gerar a predição.")

        c1, c2, c3 = st.columns(3)

        if "frequencia_percent" in aluno:
            c1.metric("Frequência", f"{aluno['frequencia_percent']:.1f}%")

        if "faltas" in aluno:
            c2.metric("Faltas", int(aluno["faltas"]))

        if "atrasos" in aluno:
            c3.metric("Atrasos", int(aluno["atrasos"]))

        c4, c5, c6 = st.columns(3)

        if "media" in aluno:
            c4.metric("Média", f"{aluno['media']:.2f}")

        if "risco_academico" in aluno:
            c5.metric("Risco acadêmico", int(aluno["risco_academico"]))

        if "valor_real" in aluno:
            c6.metric("Classe real", aluno["valor_real"])

        st.info(
            "Essas informações representam o perfil acadêmico do aluno. "
            "O modelo utiliza esses dados para estimar se ele está em situação "
            "de aprovação ou risco de reprovação."
        )

    with col_saida:
        st.markdown("#### Saída do modelo")
        st.caption("Resultado gerado pela inteligência artificial.")

        classe_prevista = aluno.get("valor_previsto", "Não disponível")
        classe_real = aluno.get("valor_real", "Não disponível")
        acertou = aluno.get("acertou", None)

        prob_reprovado = aluno.get("prob_Reprovado", None)
        prob_aprovado = aluno.get("prob_Aprovado", None)

        if classe_prevista == "Reprovado":
            st.error(f"Previsão do modelo: {classe_prevista}")
        else:
            st.success(f"Previsão do modelo: {classe_prevista}")

        if prob_reprovado is not None:
            st.metric(
                "Probabilidade de reprovação",
                f"{prob_reprovado * 100:.2f}%"
            )

            st.progress(float(prob_reprovado))

            if prob_reprovado >= 0.70:
                st.warning("Status: alto risco pedagógico")
                st.write(
                    "Recomendação: acompanhar o aluno de forma preventiva, "
                    "verificar frequência, desempenho e possíveis dificuldades."
                )
            elif prob_reprovado >= 0.40:
                st.info("Status: risco moderado")
                st.write(
                    "Recomendação: monitorar o aluno e observar evolução nas próximas avaliações."
                )
            else:
                st.success("Status: baixo risco pedagógico")
                st.write(
                    "Recomendação: manter acompanhamento regular."
                )

        if acertou is not None:
            if bool(acertou):
                st.success("O modelo acertou esta predição.")
            else:
                st.error("O modelo errou esta predição.")

        st.caption(
            f"Classe real: {classe_real} | Classe prevista: {classe_prevista}"
        )

    st.divider()

    with st.expander("Ver tabela técnica com todas as predições"):
        st.markdown("#### Classes reais, classes preditas e probabilidades")

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

        colunas_prob = [
            col for col in df_predicoes.columns if col.startswith("prob_")
        ]

        colunas_exibir = [
            col for col in colunas_prioritarias + colunas_prob
            if col in df_predicoes.columns
        ]

        st.dataframe(
            df_predicoes[colunas_exibir],
            width="stretch",
            hide_index=True,
        )

with tab3:
    st.markdown("### Download dos resultados")

    csv_predicoes = df_predicoes.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Baixar predições da amostra em CSV",
        data=csv_predicoes,
        file_name="predicoes_amostra.csv",
        mime="text/csv",
    )