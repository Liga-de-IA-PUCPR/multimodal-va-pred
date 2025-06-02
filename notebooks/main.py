import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
# Ensure config is imported early, as it handles dotenv loading.
import utils.config as config 
from utils.dataset import Affwild2GraphDataset
# For PyTorch Geometric, DataLoader is imported from torch_geometric.loader
from torch_geometric.loader import DataLoader as PyGDataLoader 
from models.GTAT.train_emotion_model import EmotionTrainer
from models.GTAT.predict_emotion import EmotionPredictor

def main():
    parser = argparse.ArgumentParser(description="Process video data into graph structures, train or predict emotions.")
    
    # Grupo mutuamente exclusivo para operações principais
    operation_group = parser.add_mutually_exclusive_group(required=True)
    operation_group.add_argument(
        "--video_ids",
        type=str,
        help="Comma-separated list of video IDs, e.g. '461,462,463'"
    )
    operation_group.add_argument(
        "--all",
        action="store_true",
        help=f"Use all videos found as subdirectories in root_dir/{config.VISUAL_FRAMES_SUBDIR_NAME}/"
    )
    operation_group.add_argument(
        "--train",
        action="store_true",
        help="Train the emotion prediction model"
    )
    operation_group.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction using a trained model"
    )
    
    # Argumentos comuns
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/raw", # Example: "data/affwild2_processed"
        help="Path to your root data folder. This folder should contain "
             f"'{config.VISUAL_FRAMES_SUBDIR_NAME}/' for frame images and "
             f"'{config.AUDIO_FILES_SUBDIR_NAME}/' for .mp4 audio source files."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1, 
        help="Batch size for PyG DataLoader. Note: PyG DataLoader handles batching of Data objects."
    )
    
    # Argumentos para treinamento
    training_group = parser.add_argument_group('Training parameters')
    training_group.add_argument(
        "--train_video_ids_csv",
        type=str,
        help="Path to CSV file containing video IDs for training"
    )
    training_group.add_argument(
        "--val_video_ids_csv",
        type=str,
        help="Path to CSV file containing video IDs for validation"
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    training_group.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    training_group.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
    )
    training_group.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save model outputs"
    )
    
    # Argumentos para predição
    prediction_group = parser.add_argument_group('Prediction parameters')
    prediction_group.add_argument(
        "--model_path",
        type=str,
        help="Path to trained model (.pth file)"
    )
    prediction_group.add_argument(
        "--test_video_ids",
        type=str,
        help="Path to CSV file containing video IDs for testing (optionally with ground truth)"
    )
    
    # Argumentos comuns para arquitetura do modelo
    model_group = parser.add_argument_group('Model architecture')
    model_group.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for model layers"
    )
    model_group.add_argument(
        "--topology_dim",
        type=int,
        default=15,
        help="Dimension for topological features"
    )
    model_group.add_argument(
        "--num_gtat_layers",
        type=int,
        default=2,
        help="Number of GTAT layers"
    )
    model_group.add_argument(
        "--gtat_heads",
        type=int,
        default=4,
        help="Number of attention heads in GTAT layers"
    )
    
    args = parser.parse_args()

    # Lógica para treinamento
    if args.train:
        if not args.train_video_ids_csv or not args.val_video_ids_csv:
            print("Error: --train_video_ids_csv and --val_video_ids_csv are required for training")
            return
        
        trainer = EmotionTrainer(
            root_dir=args.root_dir,
            train_video_ids_csv=args.train_video_ids_csv,
            val_video_ids_csv=args.val_video_ids_csv,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            topology_dim=args.topology_dim,
            num_gtat_layers=args.num_gtat_layers,
            gtat_heads=args.gtat_heads,
            dropout=args.dropout,
            output_dir=args.output_dir
        )
        trainer.run()
        return
    
    # Lógica para predição
    if args.predict:
        if not args.model_path or not args.test_video_ids:
            print("Error: --model_path and --test_video_ids are required for prediction")
            return
        
        predictor = EmotionPredictor(
            model_path=args.model_path,
            test_video_ids=args.test_video_ids,
            root_dir=args.root_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            topology_dim=args.topology_dim,
            num_gtat_layers=args.num_gtat_layers,
            gtat_heads=args.gtat_heads
        )
        predictor.run()
        return
    
    '''
    
    # Código original para processamento de vídeos
    video_ids_list = []
    if args.all:
        visual_data_parent_dir = os.path.join(args.root_dir, config.VISUAL_FRAMES_SUBDIR_NAME)
        if not os.path.isdir(visual_data_parent_dir):
            print(f"Error: Visual data directory not found: {visual_data_parent_dir}")
            print(f"Please ensure --root_dir ('{args.root_dir}') is set correctly and "
                  f"the subdirectory '{config.VISUAL_FRAMES_SUBDIR_NAME}' exists within it.")
            return

        try:
            video_ids_list = sorted([
                d for d in os.listdir(visual_data_parent_dir)
                if os.path.isdir(os.path.join(visual_data_parent_dir, d))
            ])
            if not video_ids_list:
                print(f"No video ID subdirectories found in {visual_data_parent_dir}.")
                return
        except Exception as e:
            print(f"Error listing video IDs from {visual_data_parent_dir}: {e}")
            return
    else:
        video_ids_list = [v.strip() for v in args.video_ids.split(",") if v.strip()]

    if not video_ids_list:
        print("No video IDs specified or found. Exiting.")
        return
        
    print(f"Found {len(video_ids_list)} videos to process. First few: {video_ids_list[:5]}")

    dataset = Affwild2GraphDataset(video_ids=video_ids_list, root_dir=args.root_dir)
    loader = PyGDataLoader(dataset, batch_size=args.batch_size, shuffle=False) 

    print(f"\nStarting data loading with batch size {args.batch_size}...")
    processed_batches = 0
    for batch_idx, data_batch in enumerate(loader):
        processed_batches += 1
        # data_batch is a torch_geometric.data.Batch object if batch_size > 1, 
        # or a torch_geometric.data.Data object if batch_size == 1.
        
        print(f"\n[Batch {batch_idx+1}/{len(loader)}]")
        print(f"  Batch Type: {type(data_batch)}")
        
        # Accessing video_id(s) from the batch
        if args.batch_size == 1 and hasattr(data_batch, 'video_id'):
            print(f"  Video ID: {data_batch.video_id}")
        elif args.batch_size > 1 and hasattr(data_batch, 'video_id'): 
            # If video_id was a list of strings and collated, it would be accessible.
            # PyG's default collate will make `data_batch.video_id` a list.
            print(f"  Video IDs in batch: {data_batch.video_id}")
        
        print(f"  Number of graphs in batch: {data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else 1}")
        print(f"  Node features shape (total for batch): {data_batch.x.shape}")
        print(f"  Edge index shape (total for batch): {data_batch.edge_index.shape}")
        if hasattr(data_batch, 'batch'): # batch vector mapping nodes to graphs
             print(f"  Batch vector shape: {data_batch.batch.shape}")

        # --- Placeholder for your model processing ---
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # data_batch = data_batch.to(device)
        # output = model(data_batch) 
        # --- End of placeholder ---

    if processed_batches == 0 and len(video_ids_list) > 0:
         print("\nNo batches were processed. Check dataset initialization and data paths.")
    else:
        print(f"\nFinished processing {processed_batches} batches.")
    '''
    
if __name__ == "__main__":
    main()