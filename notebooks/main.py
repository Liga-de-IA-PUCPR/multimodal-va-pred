import argparse
import os
from torch.utils.data import DataLoader
from utils.dataset import Affwild2GraphDataset

def main():
    parser = argparse.ArgumentParser(description="Run graph-based VA prediction")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video_ids",
        type=str,
        help="Comma-separated list of video IDs, e.g. '461,462,463'"
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Use all videos found under root_dir"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/raw",
        help="Path to your raw data folder"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for DataLoader"
    )
    args = parser.parse_args()

    # Determine video_ids
    if args.all:
        # assume each subfolder in root_dir is named by its video ID
        video_ids = sorted(
            int(d) for d in os.listdir(args.root_dir + "cropped_aligned/")
            if os.path.isdir(os.path.join(args.root_dir, d)) and d.isdigit()
        )
    else:
        video_ids = [int(v) for v in args.video_ids.split(",")]

    # Prepare dataset & loader
    dataset = Affwild2GraphDataset(video_ids, root_dir=args.root_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Iterate and process
    for batch_idx, graph_data in enumerate(loader):
        print(f"[Batch {batch_idx}] Graph data:", graph_data)
        # ... your downstream model / training / inference here ...

if __name__ == "__main__":
    main()