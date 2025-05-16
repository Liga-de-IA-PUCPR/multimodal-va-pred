# Framework Multimodal para Estimativa de Valence-Arousal

Este é um framework para análise e predição de emoções (valência e excitação) utilizando abordagens multimodais. O projeto implementa arquiteturas baseadas em grafos para processamento conjunto de características visuais e auditivas de vídeos.

## Estrutura do Projeto

```
multimodal-va-pred/
│
├── notebooks/                   # Notebooks e código para experimentos
│   ├── main.py                  # Script principal para experimentação
│   ├── models/                  # Modelos experimentais
│   │   ├── GTAT/                # Implementação do modelo GTAT
│   │   │   ├── GTAT.py          # Arquitetura do modelo
│   │   │   ├── train_emotion_model.py  # Treinamento
│   │   │   └── predict_emotion.py      # Inferência
│   │   └── ... (outros modelos)
│   ├── utils/                   # Utilitários para notebooks
│   │   ├── audio.py             # Processamento e extração de features de áudio
│   │   ├── backbones.py         # Redes neurais base para extração de características
│   │   ├── config.py            # Configurações e constantes do projeto
│   │   ├── dataset.py           # Classes de datasets para PyTorch Geometric
│   │   ├── graph_construction.py # Construção de grafos a partir de características multimodais
│   │   └── video.py             # Processamento e extração de features visuais
│   └── config/                  # Configurações para experimentos
│
├── src/                         # Código para produção
│   ├── main.py                  # Aplicação principal
│   └── config/                  # Configurações para produção
│
├── models/                      # Modelos pré-treinados e seus artefatos
│
├── references/                  # Artigos e documentos de referência
│
├── reports/                     # Relatórios dos experimentos
│
└── preprocessing_audio/         # Scripts de pré-processamento para áudio
```

## Instalação

1. Clone o repositório:
```bash
git clone <URL_DO_REPOSITORIO>
cd multimodal-va-pred
```

2. Instale as dependências:
```bash
pip install -r notebooks/requirements.txt
```

## Uso

### Para Experimentação (pasta notebooks)

O framework suporta diferentes modos de operação através do script `notebooks/main.py`:

#### Processamento de dados

Para processar vídeos e convertê-los em estruturas de grafos:

```bash
# Processar vídeos específicos
python notebooks/main.py --video_ids "461,462,463" --root_dir "data/raw" --batch_size 1

# Processar todos os vídeos disponíveis
python notebooks/main.py --all --root_dir "data/raw" --batch_size 1
```

#### Treinamento de Modelo

Para treinar um modelo de estimativa de emoções:

```bash
python notebooks/main.py --train \
    --train_video_ids_csv "data/splits/train_ids.csv" \
    --val_video_ids_csv "data/splits/val_ids.csv" \
    --root_dir "data/raw" \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.0005 \
    --hidden_dim 128 \
    --topology_dim 15 \
    --num_gtat_layers 2 \
    --gtat_heads 4 \
    --dropout 0.2 \
    --output_dir "outputs/gtat_model"
```

#### Previsão com Modelo Treinado

Para fazer previsões usando um modelo treinado:

```bash
python notebooks/main.py --predict \
    --model_path "outputs/gtat_model/best_model.pth" \
    --test_video_ids "data/splits/test_ids.csv" \
    --root_dir "data/raw" \
    --batch_size 1 \
    --output_dir "predictions/gtat_model"
```

### Parâmetros do Modelo

Os parâmetros abaixo são comuns para treinamento e previsão, e definem a arquitetura do modelo:

- `--hidden_dim`: Dimensão oculta para as camadas do modelo
- `--topology_dim`: Dimensão das características topológicas dos dados
- `--num_gtat_layers`: Número de camadas GTAT usadas no modelo
- `--gtat_heads`: Número de cabeças de atenção em cada camada GTAT

### Para Produção (pasta src)

A interface de produção está sendo desenvolvida na pasta `src/`. Consulte a documentação específica para mais detalhes.

## Adicionando Novos Modelos

O framework foi projetado para facilitar a adição de novos modelos. Para adicionar um novo modelo:

1. Crie uma nova pasta em `notebooks/models/` com o nome do seu modelo
2. Implemente no mínimo:
   - A arquitetura do modelo
   - Um script de treinamento
   - Um script de previsão

Consulte a implementação do modelo GTAT como exemplo.

## Licença

[Especificar a licença do projeto]
