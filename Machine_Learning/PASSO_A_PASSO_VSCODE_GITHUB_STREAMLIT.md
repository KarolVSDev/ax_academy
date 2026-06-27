# Passo a passo — VSCode, ambiente virtual, Streamlit e GitHub

## 1. Abrir a pasta no VSCode

1. Baixar e extrair a pasta do projeto.
2. Abrir o VSCode.
3. Clicar em **File > Open Folder**.
4. Selecionar a pasta `projeto_risco_pedagogico_streamlit`.

## 2. Colocar os datasets no lugar correto

Dentro do projeto, colocar os arquivos CSV em:

```text
data/processed/
```

Os nomes precisam ficar exatamente assim:

```text
dataset_treino_validacao_padrao_realista.csv
dataset_teste_final_padrao_realista.csv
```

## 3. Criar ambiente virtual

No terminal do VSCode, executar:

```powershell
py -m venv .venv
```

Ativar o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se der erro de permissão no PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Depois ativar de novo:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 4. Instalar as bibliotecas

Com o ambiente virtual ativado:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Treinar o modelo

```powershell
python -m src.train
```

Esse comando cria:

```text
models/modelo_vencedor.joblib
models/metadata_modelo.json
```

## 6. Abrir a aplicação Streamlit

```powershell
streamlit run app.py
```

O navegador abrirá uma página local da aplicação.

## 7. Subir para o GitHub

Criar um repositório vazio no GitHub e executar:

```powershell
git init
git add .
git commit -m "Cria MVP Streamlit de risco pedagogico"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

Se o remote já existir:

```powershell
git remote set-url origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

## 8. Explicação curta para o pitch

"Este sistema transforma o modelo de Machine Learning em uma aplicação prática. O problema de negócio é identificar alunos com risco de reprovação. O conjunto de teste foi mantido separado do treinamento. Durante a apresentação, a banca escolhe o percentual de dados inéditos que deseja testar, a aplicação sorteia a amostra, executa a predição e apresenta as métricas de desempenho em tempo real."
