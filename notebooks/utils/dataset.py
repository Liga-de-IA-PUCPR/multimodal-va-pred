from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import os
import pandas as pd

from utils.audio import extract_audio_features_for_video
from utils.video import extract_visual_features
from utils.graph_construction import build_video_graph
from utils.config import (
    NUM_FRAMES, 
    VISUAL_FRAMES_SUBDIR_NAME, 
    AUDIO_FILES_SUBDIR_NAME,
    AUDIO_MEL_N_MELS, 
    VISUAL_BACKBONE_OUT_DIM
)

class Affwild2GraphDataset(Dataset):
    def __init__(self, video_ids, root_dir: str, annotations=None):
        """
        Dataset para grafos de vídeo do Aff-Wild2 para reconhecimento de emoções.
        
        Args:
            video_ids: Lista de IDs de vídeo a serem processados
            root_dir: Diretório raiz contendo subdiretórios de dados
            annotations_df: (Opcional) DataFrame pandas contendo anotações de valência e excitação
                            O DataFrame deve ter colunas 'video_id', 'valence' e 'arousal'
        """
        self.video_ids = video_ids
        self.root_dir = root_dir
        self.annotations = annotations

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = str(self.video_ids[idx])
        
        visual_frames_folder = os.path.join(self.root_dir, VISUAL_FRAMES_SUBDIR_NAME, video_id)
        
        # --- Determine audio source path with .mp4 and .avi fallback ---
        audio_dir = os.path.join(self.root_dir, AUDIO_FILES_SUBDIR_NAME)
        mp4_path = os.path.join(audio_dir, f"{video_id}.mp4")
        avi_path = os.path.join(audio_dir, f"{video_id}.avi")

        audio_source_path = None
        if os.path.exists(mp4_path):
            audio_source_path = mp4_path
        elif os.path.exists(avi_path):
            audio_source_path = avi_path
            print(f"Info: Using .avi audio source for video {video_id}: {avi_path}")
        else:
            # If neither exists, audio extraction will handle the error and return zeros.
            # We pass a placeholder (e.g., expected mp4_path) to trigger that handling.
            audio_source_path = mp4_path 
            print(f"Warning: No .mp4 or .avi audio source found for video {video_id} at expected paths:\n"
                  f"  {mp4_path}\n  {avi_path}\n"
                  f"  Proceeding with zero audio features.")
        # --- End audio source path determination ---

        # Initialize valence and arousal
        valence_tensor = None
        arousal_tensor = None
        csv_path = None

        # Extrair csv_path correspondente ao video_id usando a lista de annotations
        if self.annotations is not None:
            for path in self.annotations:
                if os.path.splitext(os.path.basename(path))[0] == video_id:
                    csv_path = path
                    break

        # Ler o CSV e extrair as colunas "valence" e "arousal"
        if csv_path is not None:
            try:
                df = pd.read_csv(csv_path)
                if not df.empty and 'valence' in df.columns and 'arousal' in df.columns:
                    # Se houver múltiplas linhas, calcula a média dos valores
                    valence = df['valence']
                    arousal = df['arousal']
                    valence_tensor = torch.tensor(valence, dtype=torch.float)
                    arousal_tensor = torch.tensor(arousal, dtype=torch.float)
                elif df.empty:
                    print(f"Warning: Annotation file {csv_path} for video {video_id} is empty.")
                else:
                    print(f"Warning: Columns 'valence' or 'arousal' not found in {csv_path} for video {video_id}.")
            except Exception as e:
                print(f"Error reading or processing annotation file {csv_path} for video {video_id}: {e}")
        else:
            if self.annotations is not None: # Only warn if annotations were expected
                 print(f"Warning: No annotation file found for video {video_id}.")


        visual_features = extract_visual_features(visual_frames_folder, num_frames_to_extract=NUM_FRAMES)
        audio_features = extract_audio_features_for_video(audio_source_path) # Handles non-existent path
        
        visual_features = visual_features.cpu()
        audio_features = audio_features.cpu()

        if visual_features.size(0) != NUM_FRAMES:
            print(f"Warning: Visual features for {video_id} have {visual_features.size(0)} frames, expected {NUM_FRAMES}. Interpolating.")
            if visual_features.size(0) == 0:
                visual_features = torch.zeros(NUM_FRAMES, VISUAL_BACKBONE_OUT_DIM)
            else:
                visual_features = visual_features.transpose(0,1).unsqueeze(0)
                visual_features = F.interpolate(
                    visual_features, size=NUM_FRAMES, mode='linear', align_corners=False
                ).squeeze(0).transpose(0,1)

        if audio_features.size(0) != NUM_FRAMES:
            print(f"Warning: Audio features for {video_id} have {audio_features.size(0)} frames, expected {NUM_FRAMES}. Interpolating.")
            if audio_features.size(0) == 0:
                audio_features = torch.zeros(NUM_FRAMES, AUDIO_MEL_N_MELS)
            else:
                audio_features = audio_features.transpose(0, 1).unsqueeze(0)
                audio_features = F.interpolate(
                    audio_features, size=NUM_FRAMES, mode='linear', align_corners=False
                ).squeeze(0).transpose(0, 1)
        
        graph_data = build_video_graph(visual_features, audio_features, valence_tensor, arousal_tensor)
        graph_data.video_id = video_id
        
        return graph_data