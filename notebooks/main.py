import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
# Ensure config is imported early, as it handles dotenv loading.
import utils.config as config 
from utils.dataset import Affwild2GraphDataset
# For PyTorch Geometric, DataLoader is imported from torch_geometric.loader
from torch_geometric.loader import DataLoader as PyGDataLoader 

def main():
    parser = argparse.ArgumentParser(description="Process video data into graph structures.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video_ids",
        type=str,
        help="Comma-separated list of video IDs, e.g. '461,462,463'"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help=f"Use all videos found as subdirectories in root_dir/{config.VISUAL_FRAMES_SUBDIR_NAME}/"
    )
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
    args = parser.parse_args()

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

if __name__ == "__main__":
    main()