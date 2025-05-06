# from fusion_gnn import GNNFusion

# model = GNNFusion(in_audio_dim=audio_feat_dim,
#                   in_visual_dim=visual_feat_dim).to(device)

# # In training loop:
# data = build_graph_batch(audio_feats, visual_feats)  # your helper
# pred = model(data)  # [batch_size, 2]
# loss = criterion(pred, labels)  # regression on valence/arousal