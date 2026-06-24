# ReciclaAI — Classificador de Resíduos

Pipeline completo: organizar dataset → treinar (ResNet18 + transfer learning) → servir um front-end para classificar imagens.

## 1. Estrutura de arquivos

```
reciclaai/
├── split_dataset.py     # organiza o dataset do Kaggle em train/test
├── treino.py             # treina o modelo
├── app.py                 # backend Flask (carrega o modelo e expõe /predict)
├── templates/
│   └── index.html         # front-end de upload
├── requirements.txt
└── model/                 # criado automaticamente, guarda reciclaai.pth
```

## 2. Ambiente

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```

**GPU (recomendado):** se sua máquina tem uma GPU NVIDIA, instale o PyTorch com suporte a CUDA *antes* de rodar o requirements.txt, seguindo o seletor em https://pytorch.org/get-started/locally/ (escolha a versão do CUDA da sua placa). Sem isso, o `torch` instala a versão CPU-only e o treino fica bem mais lento.

Confirme que a GPU foi detectada:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 3. Organizar o dataset do Kaggle

O dataset que você baixou tem uma pasta por classe (cardboard, glass, metal, paper, plastic, trash), sem divisão train/test. O `treino.py` espera `dataset/train/<classe>` e `dataset/test/<classe>`, então rode:

```bash
python split_dataset.py --source "C:\caminho\para\o\dataset\baixado" --dest dataset --test-size 0.2
```

Isso copia ~80% das imagens de cada classe para `dataset/train/` e ~20% para `dataset/test/`. Use `--move` em vez de copiar se quiser economizar espaço em disco.

## 4. Treinar

```bash
python treino.py
```

Otimizações já aplicadas no script (em relação ao original) para rodar mais rápido sem mudar a arquitetura:
- `torch.backends.cudnn.benchmark = True` quando há GPU (as imagens são sempre 224x224, então o cuDNN pode escolher o melhor algoritmo de convolução).
- `DataLoader` com `num_workers=4` e `pin_memory=True`: carrega os próximos lotes em paralelo enquanto a GPU treina o lote atual, em vez de ficar esperando o disco.
- **Mixed precision (AMP)** com `torch.amp.autocast` + `GradScaler`: faz a maior parte das contas em float16 na GPU, normalmente ~1.5–2x mais rápido e usando menos memória, sem perda perceptível de acurácia.
- `non_blocking=True` ao mover tensores para a GPU.

Outras dicas que dependem da sua máquina:
- Se tiver bastante VRAM livre, aumente `BATCH_SIZE` em `treino.py` (ex.: 64 ou 128) — acelera o treino ao custo de mais memória.
- No Windows, se o `DataLoader` travar ou der erro de multiprocessing, mude `NUM_WORKERS` para `0`.
- Como só a camada `fc` é treinada (o resto do ResNet18 está congelado), 8 épocas costumam ser suficientes; acompanhe `Test Acc` no log e pare antes se ela parar de melhorar.

Ao final, o modelo é salvo em `model/reciclaai.pth`.

## 5. Rodar o front-end

```bash
python app.py
```

Acesse **http://localhost:5000** no navegador. Arraste ou selecione uma foto de um resíduo e clique em "Classificar" — o backend roda a inferência com o modelo treinado e devolve a categoria prevista com o nível de confiança de cada classe.

> O `app.py` precisa que `model/reciclaai.pth` já exista (passo 4) antes de iniciar.
