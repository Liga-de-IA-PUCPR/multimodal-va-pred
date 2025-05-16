# GTAT - Modelo de Reconhecimento de Emoções baseado em Grafos

Este diretório contém a implementação do modelo Graph Topology Aware Transformer (GTAT) para reconhecimento de emoções em vídeos. O modelo utiliza uma abordagem baseada em grafos para processamento conjunto de características visuais e auditivas.

## Visão Geral

O modelo GTAT utiliza uma abordagem baseada em grafos para processar dados de vídeo, onde cada quadro é representado como um nó no grafo, e as conexões temporais entre quadros são representadas como arestas. O modelo é capaz de prever dois valores emocionais principais:

- **Valência**: A positividade ou negatividade da emoção (variando de negativa a positiva)
- **Excitação**: A intensidade da emoção (variando de calma a excitada)

## Arquitetura GTAT

A arquitetura GTAT (Graph Topology Aware Transformer) consiste em:

1. **Extrator de Características Topológicas**: Extrai características de topologia dos nós
2. **Camadas GTAT**: Processa nós do grafo usando atenção que considera tanto características do nó quanto topologia
3. **Pooling Global**: Combina informações de todos os nós do grafo
4. **Camadas de Saída**: Prevê os valores de valência e excitação

## Estrutura dos Arquivos

```
GTAT/
├── GTAT.py                 # Implementação da arquitetura do modelo
├── train_emotion_model.py  # Classe EmotionTrainer para treinamento
├── predict_emotion.py      # Classe EmotionPredictor para previsão
└── README.md               # Este arquivo
```

## Como Usar

O modelo GTAT pode ser executado através do script principal `main.py` na pasta notebooks.

### Treinamento do Modelo

Para treinar o modelo GTAT:

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

O arquivo CSV de IDs de vídeo deve conter as seguintes colunas:
- `video_id`: ID único do vídeo
- `valence`: Valor de valência (alvo)
- `arousal`: Valor de excitação (alvo)

### Previsão com Modelo Treinado

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

Os parâmetros principais que definem a arquitetura do modelo GTAT são:

- `--hidden_dim`: Dimensão oculta para as camadas do modelo (padrão: 128)
- `--topology_dim`: Dimensão das características topológicas dos dados (padrão: 15)
- `--num_gtat_layers`: Número de camadas GTAT usadas no modelo (padrão: 2)
- `--gtat_heads`: Número de cabeças de atenção em cada camada GTAT (padrão: 4)
- `--dropout`: Taxa de dropout para regularização (padrão: 0.2)

## Avaliação do Modelo

O modelo é avaliado usando o Coeficiente de Correlação de Concordância (CCC), uma métrica comum em reconhecimento de emoções que mede a concordância entre valores previstos e valores verdadeiros.

## Artefatos Gerados

### Após o treinamento

- Modelo salvo (`best_model.pth`): O modelo com melhor desempenho durante a validação
- Histórico de treinamento (`training_history.csv`): Métricas de treinamento e validação por época
- Curvas de treinamento (`training_curves.png`): Visualização do progresso de treinamento

### Após a previsão

- Arquivo CSV com previsões (`predictions.csv`): Contém as previsões para cada vídeo de teste
- Gráficos comparativos: Comparação visual entre valores previstos e reais (quando disponíveis)

## Customização

Para customizar o modelo GTAT:

1. Modifique os hiperparâmetros através das opções de linha de comando
2. Para alterações mais profundas na arquitetura, modifique o arquivo `GTAT.py`
3. Para alterar o processo de treinamento, modifique `train_emotion_model.py`
4. Para alterar o processo de previsão, modifique `predict_emotion.py` 