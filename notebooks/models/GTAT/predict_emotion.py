import os
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import Affwild2GraphDataset
from notebooks.models.GTAT.GTAT import GCATopo
from torch_geometric.loader import DataLoader

class EmotionPredictor:
    """
    Classe para previsão de emoções utilizando o modelo GTAT.
    
    Parâmetros:
        model_path (str): Caminho para o modelo treinado (.pth).
        test_video_ids (str): Arquivo CSV contendo IDs de vídeo para teste (opcionalmente com ground truth).
        root_dir (str): Diretório raiz que contém os dados brutos.
        output_dir (str): Diretório onde as previsões serão salvas.
        batch_size (int): Tamanho do lote para realização da inferência.
        hidden_dim (int): Dimensão oculta para as camadas do modelo.
        topology_dim (int): Dimensão das características topológicas dos dados.
        num_gtat_layers (int): Número de camadas GTAT usadas no modelo.
        gtat_heads (int): Número de cabeças de atenção em cada camada GTAT.
    """
    def __init__(self, model_path, test_video_ids, root_dir="data/raw", output_dir="predictions",
                 batch_size=1, hidden_dim=128, topology_dim=15, num_gtat_layers=2, gtat_heads=4):
        self.model_path = model_path
        self.test_video_ids = test_video_ids
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.topology_dim = topology_dim
        self.num_gtat_layers = num_gtat_layers
        self.gtat_heads = gtat_heads

    def _plot_predictions(self, video_id, valence_true, arousal_true, valence_pred, arousal_pred):
        # Plotar previsões e valores reais para um vídeo específico
        plt.figure(figsize=(12, 6))
        # Valência
        plt.subplot(2, 1, 1)
        plt.plot(valence_true, label='Real', color='blue')
        plt.plot(valence_pred, label='Previsto', color='red')
        plt.xlabel('Quadros')
        plt.ylabel('Valência')
        plt.title(f'Vídeo {video_id} - Valência')
        plt.grid(True)
        plt.legend()
        # Excitação
        plt.subplot(2, 1, 2)
        plt.plot(arousal_true, label='Real', color='blue')
        plt.plot(arousal_pred, label='Previsto', color='red')
        plt.xlabel('Quadros')
        plt.ylabel('Excitação')
        plt.title(f'Vídeo {video_id} - Excitação')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'prediction_{video_id}.png'))
        plt.close()

    def _concordance_correlation_coefficient(self, y_true, y_pred):
        """
        Calcular CCC entre valores verdadeiros e previstos
        """
        y_true_mean = np.mean(y_true)
        y_pred_mean = np.mean(y_pred)
        y_true_var = np.var(y_true, ddof=0)
        y_pred_var = np.var(y_pred, ddof=0)
        covariance = np.mean((y_true - y_true_mean) * (y_pred - y_pred_mean))
        ccc = (2 * covariance) / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2)
        return ccc

    def run(self):
        # Criar diretório de saída
        os.makedirs(self.output_dir, exist_ok=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {device}")
        
        # Carregar IDs de vídeo para teste
        test_data = pd.read_csv(self.test_video_ids)
        test_video_ids_list = test_data['video_id'].tolist()
        
        # Verificar se há ground truth
        has_ground_truth = 'valence' in test_data.columns and 'arousal' in test_data.columns
        
        # Criar conjunto de dados e carregador
        test_dataset = Affwild2GraphDataset(
            video_ids=test_video_ids_list, 
            root_dir=self.root_dir,
            annotations_df=test_data if has_ground_truth else None
        )
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Inicializar e carregar o modelo
        model = GCATopo(
            hidden_dim=self.hidden_dim,
            topology_dim=self.topology_dim,
            num_gtat_layers=self.num_gtat_layers,
            gtat_heads=self.gtat_heads
        ).to(device)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()
        
        results = {
            'video_id': [],
            'valence_pred': [],
            'arousal_pred': []
        }
        if has_ground_truth:
            results['valence_true'] = []
            results['arousal_true'] = []
            
        all_valence_true = []
        all_arousal_true = []
        all_valence_pred = []
        all_arousal_pred = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Fazendo previsões"):
                data = data.to(device)
                # Previsão do modelo
                valence_pred, arousal_pred = model(data)
                video_id = data.video_id[0] if self.batch_size > 1 else data.video_id
                valence_pred_np = valence_pred.cpu().numpy().flatten()
                arousal_pred_np = arousal_pred.cpu().numpy().flatten()
                
                results['video_id'].append(video_id)
                results['valence_pred'].append(valence_pred_np[0])
                results['arousal_pred'].append(arousal_pred_np[0])
                
                all_valence_pred.extend(valence_pred_np)
                all_arousal_pred.extend(arousal_pred_np)
                
                if has_ground_truth:
                    valence_true = data.valence.cpu().numpy().flatten()
                    arousal_true = data.arousal.cpu().numpy().flatten()
                    results['valence_true'].append(valence_true[0])
                    results['arousal_true'].append(arousal_true[0])
                    
                    all_valence_true.extend(valence_true)
                    all_arousal_true.extend(arousal_true)
                    
                    if self.batch_size == 1:
                        self._plot_predictions(video_id, valence_true, arousal_true,
                                               valence_pred_np, arousal_pred_np)
        
        if has_ground_truth:
            valence_ccc = self._concordance_correlation_coefficient(np.array(all_valence_true), np.array(all_valence_pred))
            arousal_ccc = self._concordance_correlation_coefficient(np.array(all_arousal_true), np.array(all_arousal_pred))
            print(f"Métricas de desempenho:")
            print(f"  Valência CCC: {valence_ccc:.4f}")
            print(f"  Excitação CCC: {arousal_ccc:.4f}")
            print(f"  Média CCC: {(valence_ccc + arousal_ccc) / 2:.4f}")
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(self.output_dir, 'predictions.csv'), index=False)
        print(f"Previsões concluídas e salvas em {self.output_dir}")
        return results_df
