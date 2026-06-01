# MVP — Classificação de Risco Pedagógico com Streamlit

Aplicação desenvolvida para demonstrar um modelo de Machine Learning capaz de classificar alunos como **Aprovado** ou **Reprovado**, com foco na identificação de risco pedagógico.

## Objetivo do MVP

A banca escolhe dinamicamente um percentual do conjunto de teste final. A aplicação sorteia uma amostra aleatória, executa a inferência do modelo e exibe:

- classes reais e preditas;
- métricas de desempenho;
- matriz de confusão;
- tabela de resultados;
- download das predições.

## Estrutura do projeto

```text
projeto_risco_pedagogico_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml
├── data/
│   └── processed/
│       ├── dataset_treino_validacao_padrao_realista.csv
│       └── dataset_teste_final_padrao_realista.csv
├── models/
│   ├── modelo_vencedor.joblib
│   └── metadata_modelo.json
├── notebooks/
│   └── notebook_original.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── features.py
    ├── train.py
    └── predict.py
```

## Arquivos de dados esperados

Coloque estes dois arquivos dentro de `data/processed/`:

```text
dataset_treino_validacao_padrao_realista.csv
dataset_teste_final_padrao_realista.csv
```

O conjunto de teste final deve permanecer separado do treino e da validação.

## Como executar no VSCode

No terminal PowerShell, dentro da pasta do projeto:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

Se o PowerShell bloquear a ativação do ambiente virtual, execute uma vez:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Depois tente ativar novamente:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Como subir para o GitHub

```powershell
git init
git add .
git commit -m "Cria MVP Streamlit de classificação de risco pedagógico"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

Se aparecer erro `remote origin already exists`, use:

```powershell
git remote set-url origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

## Observação sobre dados e modelo

Por padrão, o `.gitignore` não envia os CSVs nem o modelo treinado para o GitHub. Isso evita subir arquivos grandes ou dados sensíveis. Para apresentar localmente, mantenha os arquivos dentro das pastas indicadas.
