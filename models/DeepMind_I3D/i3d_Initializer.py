import cv2
import numpy as np
import torch
from torchvision import transforms
from pytorch_i3d_new import InceptionI3d  # Supondo que você tenha uma implementação do I3D disponível


class I3DModelWrapper:
    def __init__(self, pretrained_path: str, num_classes: int = 400, in_channels: int = 3):
        """
        Inicializa o modelo I3D, carrega os pesos pré-treinados, exibe uma mensagem de carregamento,
        move o modelo para o dispositivo apropriado (GPU, se disponível) e coloca o modelo no modo de avaliação.

        Args:
            pretrained_path (str): Caminho para os pesos pré-treinados do modelo I3D.
            num_classes (int): Número de classes do modelo.
            in_channels (int): Número de canais de entrada (geralmente 3 para RGB).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Carregando pesos do arquivo: {pretrained_path}")
        # Carrega o modelo I3D pré-treinado no Kinetics
        self.model = InceptionI3d(num_classes=num_classes, in_channels=in_channels)
        state_dict = torch.load(pretrained_path)
        self.model.load_state_dict(state_dict)
        # Mostra mensagem de sucesso no carregamento dos pesos
        print(f"[INFO] Pesos carregados com sucesso. Dimensões dos pesos: {len(state_dict)} itens encontrados.")

        # Substitui a camada de logits pela identidade para obter os vetores de features
        self.model.logits = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Modelo movido para o dispositivo: {self.device}")

    def I_3D(self, video: torch.Tensor) -> torch.Tensor:
        """
        Realiza o encaminhamento do tensor de vídeo através do modelo I3D para extrair vetores de características.

        Args:
            video (torch.Tensor): Tensor de vídeo pré-processado no formato
                                  [batch_size, 3, num_frames, height, width].

        Returns:
            torch.Tensor: Vetores de características extraídos.
        """
        video = video.to(self.device)
        print("[DEBUG] Executando extração de características com I3D...")
        with torch.no_grad():
            features = self.model(video)
        print("[DEBUG] Extração concluída. Processando saída do segmento.")
        return features

    def preprocess_video(self, video_path: str, num_frames: int = 64, resize: tuple = (224, 224)) -> torch.Tensor:
        """
        Pré-processa o vídeo lendo-o, extraindo um número fixo de frames,
        redimensionando cada frame, convertendo-os para tensor e empilhando em um único tensor.

        Parâmetros:
            video_path (str): Caminho para o arquivo de vídeo.
            num_frames (int): Número de frames a serem extraídos.
            resize (tuple): Tupla (width, height) para redimensionamento dos frames.

        Retorna:
            torch.Tensor: Tensor representando o vídeo com formato (C, T, H, W),
                          onde C é canais, T é frames, H é altura e W é largura.
        """
        print(f"[INFO] Iniciando preprocessamento do vídeo: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] Total de frames no vídeo: {total_frames}")

        if total_frames < num_frames:
            frame_indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frame_idx_set = set(frame_indices)
        current_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if current_index in frame_idx_set:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)
                frames.append(frame)
            current_index += 1
        cap.release()
        print(f"[INFO] Extração concluída dos frames selecionados.")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        tensor_frames = [transform(frame) for frame in frames]
        video_tensor = torch.stack(tensor_frames)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        return video_tensor

    def extract_features_for_segments(self, video_path: str, segment_duration: float = 5.0,
                                      resize: tuple = (224, 224)) -> list:
        """
        Extrai vetores de características para cada segmento do vídeo.
        O vídeo é dividido em segmentos com a duração especificada e cada segmento é processado
        através do modelo I3D para extração dos vetores de features.

        Args:
            video_path (str): Caminho para o arquivo de vídeo.
            segment_duration (float): Duração de cada segmento em segundos.
            resize (tuple): Tupla (width, height) para redimensionamento dos frames.

        Returns:
            List[torch.Tensor]: Lista onde cada elemento é o vetor de características correspondendo
                                a um segmento do vídeo.
        """
        print(f"[INFO] Iniciando extração de features por segmento do vídeo: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # fallback se FPS não for definido
        segment_frame_count = int(fps * segment_duration)
        print(f"[INFO] FPS: {fps}. Número de frames por segmento (duração {segment_duration}s): {segment_frame_count}")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        segments_features = []
        segment_idx = 0

        while True:
            segment_frames = []
            for i in range(segment_frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, resize)
                segment_frames.append(transform(frame))
            if not segment_frames:
                break

            video_tensor = torch.stack(segment_frames)
            video_tensor = video_tensor.permute(1, 0, 2, 3)
            video_tensor = video_tensor.unsqueeze(0)

            print(f"[DEBUG] Processando segmento {segment_idx + 1} com {len(segment_frames)} frames...")
            features = self.I_3D(video_tensor)
            segments_features.append(features.squeeze(0))
            segment_idx += 1

        cap.release()
        print(f"[INFO] Extração concluída para {segment_idx} segmentos.")
        return segments_features


# Exemplo de uso da classe
if __name__ == "__main__":
    pretrained_path = "./pretrained/rgb_imagenet.pt"  # Substitua com o caminho dos seus pesos
    #Teste com uma Imagem da RECOLA
    video_path = "/Users/lfbf/Library/CloudStorage/OneDrive-GrupoMarista/RECOLA/RECOLA-Video-recordings/P16.mp4"  # Substitua com o caminho do seu vídeo

    print("[INFO] Inicializando I3DModelWrapper...")
    model_wrapper = I3DModelWrapper(pretrained_path=pretrained_path)

    # Exemplo de preprocessamento do vídeo
    video_tensor = model_wrapper.preprocess_video(video_path)
    print(f"[INFO] Video tensor shape: {video_tensor.shape}")

    # Extração de features por segmentos
    segments_features = model_wrapper.extract_features_for_segments(video_path)
    for idx, feature in enumerate(segments_features):
        print(f"[INFO] Segmento {idx + 1} - forma do feature: {feature.shape}")