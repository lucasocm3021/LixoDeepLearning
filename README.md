# ReciclaAI ♻️

Classificador de resíduos recicláveis usando *deep learning*. O modelo (ResNet18 com *transfer learning*) é treinado para reconhecer 6 categorias de resíduo — **papelão, vidro, metal, papel, plástico e rejeito** — e um front-end web simples permite enviar uma foto e ver a previsão na hora.

## Sumário

- [Sobre o projeto](#sobre-o-projeto)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Pré-requisitos](#pré-requisitos)
- [Passo a passo](#passo-a-passo)
  - [1. Clonar o repositório](#1-clonar-o-repositório)
  - [2. Criar o ambiente virtual](#2-criar-o-ambiente-virtual)
  - [3. Instalar as dependências](#3-instalar-as-dependências)
  - [4. Baixar e organizar o dataset](#4-baixar-e-organizar-o-dataset)
  - [5. Treinar o modelo](#5-treinar-o-modelo)
  - [6. Rodar o front-end](#6-rodar-o-front-end)
- [Solução de problemas](#solução-de-problemas)

## Sobre o projeto

O pipeline tem três etapas:

1. **`split_dataset.py`** organiza as imagens baixadas do Kaggle na estrutura de pastas que o treino espera (`train/` e `test/`).
2. **`treino.py`** treina uma ResNet18 pré-treinada (congelando as camadas convolucionais e treinando só a camada final) e salva o modelo em `model/reciclaai.pth`.
3. **`app.py`** carrega esse modelo e expõe uma página web (`templates/index.html`) onde qualquer pessoa pode arrastar uma foto e ver a categoria prevista, com o nível de confiança de cada classe.

## Estrutura do repositório

```
ReciclaAI/
├── split_dataset.py     # organiza o dataset do Kaggle em train/test
├── treino.py             # treina o modelo
├── app.py                 # backend Flask (carrega o modelo e expõe /predict)
├── templates/
│   └── index.html         # front-end de upload
├── requirements.txt
├── .gitignore
└── README.md
```

> As pastas `dataset/`, `model/` e `venv/` **não fazem parte do repositório** (estão no `.gitignore`) — você as gera localmente seguindo os passos abaixo.

## Pré-requisitos

- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads)
- Conta gratuita no [Kaggle](https://www.kaggle.com/) para baixar o dataset
- *(opcional, recomendado)* GPU NVIDIA com driver atualizado — o treino roda em CPU também, só é mais lento

## Passo a passo

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/ReciclaAI.git
cd ReciclaAI
```

### 2. Criar o ambiente virtual

O ambiente virtual isola as dependências deste projeto do resto da sua máquina.

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

O prompt deve passar a mostrar `(venv)` no início da linha quando o ambiente estiver ativo. Repita a ativação a cada nova sessão do terminal.

### 3. Instalar as dependências

**Se você tem GPU NVIDIA** (recomendado — o treino fica bem mais rápido):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flask pillow
```

**Se você não tem GPU (ou não sabe):**
```bash
pip install -r requirements.txt
```

> ⚠️ No Windows e macOS, `pip install torch` sozinho **sempre instala a versão CPU-only**, mesmo que você tenha uma GPU NVIDIA. Por isso o comando com `--index-url` acima é necessário para usar a GPU.

Confirme se a GPU foi detectada:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Deve aparecer `True` no final. Se aparecer `False` mesmo tendo GPU, veja a seção [Solução de problemas](#solução-de-problemas).

### 4. Baixar e organizar o dataset

Baixe o dataset **[Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)** no Kaggle (são ~2.500 imagens em 6 pastas: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) e extraia o `.zip` em algum lugar da sua máquina.

Depois, rode o script de organização para dividir em treino/teste:
```bash
python split_dataset.py --source "C:\caminho\para\o\dataset\extraido" --dest dataset --test-size 0.2
```

Isso cria a pasta `dataset/` (ignorada pelo Git) com `train/` e `test/`, cada uma com as 6 subpastas de classe.

### 5. Treinar o modelo

```bash
python treino.py
```

O treino imprime a acurácia de treino e teste a cada época e, ao final, salva o modelo em `model/reciclaai.pth`. Em GPU costuma levar poucos minutos; em CPU pode levar bem mais tempo dependendo do tamanho do dataset.

### 6. Rodar o front-end

```bash
python app.py
```

Acesse **http://localhost:5000** no navegador. Arraste ou selecione a foto de um resíduo e clique em **Classificar** — o backend roda a inferência com o modelo treinado e mostra a categoria prevista com o nível de confiança de cada classe.

> O `app.py` só funciona depois do passo 5 — ele procura o arquivo `model/reciclaai.pth`.

## Solução de problemas

**`ModuleNotFoundError: No module named 'torch'`**
O ambiente virtual está ativo (`(venv)` aparece no prompt) mas as dependências não foram instaladas nele. Repita o passo 3 com o venv ativo.

**`torch.cuda.is_available()` retorna `False` mesmo com GPU NVIDIA**
Você instalou a build CPU-only por engano. Reinstale apontando para o índice certo:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
Se `nvidia-smi` (no terminal) não reconhecer o comando, sua máquina não tem GPU NVIDIA disponível — o projeto funciona em CPU, só mais lento.

**`FileNotFoundError: Modelo não encontrado em model/reciclaai.pth`**
Você ainda não rodou o passo 5 (`python treino.py`), ou rodou em outra pasta. O `app.py` precisa ser executado na raiz do projeto.

**`FileNotFoundError` ao rodar `treino.py` (pasta `dataset/train` não existe)**
Volte ao passo 4 — o dataset baixado do Kaggle precisa passar pelo `split_dataset.py` antes, já que ele vem sem divisão treino/teste.
