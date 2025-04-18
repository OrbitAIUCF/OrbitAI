import torch
from GATDraftModel import OrbitGNN

# 1) Load graphs
graphs = OrbitGNN.build_all_graphs_from_csv("orbits_with_velocity_and_labels.csv")

# 2) Apply your handcrafted attention
graphs = OrbitGNN.apply_hardcoded_attention(graphs)

# # 3) (Optional) if you later want model‑based attention, replace data.attn:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = OrbitGNN(use_edge_embedding=True).to(device).eval()

# for g in graphs:
#     x   = g.x.to(device)
#     ei  = g.edge_index.to(device)
#     ea  = g.edge_attr.to(device)
#     with torch.no_grad():
#         _ = model(x, ei, ea)
#         g.attn = model.attn_weights.cpu().tolist()

# 4) Export everything
OrbitGNN.export_graph_list_to_json(graphs, "exported_frames")

print("JSON export complete! Check the `exported_frames/` folder.")
