import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import random

# -------------------------------
# Graph Visualization (2D)
# -------------------------------
def visualize_graph(data):
    G = nx.DiGraph()
    positions = data.x[:, :3].numpy()  # use x, y, z as node positions
    sat_ids = data.sat_ids

    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos, label=sat_ids[i])

    edge_list = data.edge_index.t().tolist()
    distances = data.edge_attr.tolist()
    for (i, j), d in zip(edge_list, distances):
        G.add_edge(i, j, weight=round(d, 1))

    pos_dict = {i: positions[i][:2] for i in range(len(positions))}  # x-y projection
    label_pos = {i: (positions[i][0] + random.uniform(1, 10),
                    positions[i][1] + random.uniform(1, 10)) for i in range(len(positions))}  # offset labels

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos=pos_dict, with_labels=True,
            labels={i: sat_ids[i] for i in range(len(sat_ids))},
            node_color='skyblue', edge_color='gray', node_size=500, font_size=10)
    nx.draw_networkx_labels(G, pos=label_pos, labels={i: sat_ids[i] for i in range(len(sat_ids))}, font_size=10)

    # Draw edge labels for distances
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos=pos_dict, edge_labels=edge_labels, font_color='red')

    plt.title("Satellite Interaction Graph (x-y plane)")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True)
    plt.show()

# -------------------------------
# 3D Visualization of Satellite Graph
# -------------------------------
def visualize_graph_3d(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    positions = data.x[:, :3].numpy()
    sat_ids = data.sat_ids

    for i, (x, y, z) in enumerate(positions):
        offset_x = x + random.uniform(1, 10)
        offset_y = y + random.uniform(1, 10)
        offset_z = z + random.uniform(1, 10)
        ax.scatter(x, y, z, label=sat_ids[i], s=60)
        ax.text(offset_x, offset_y, offset_z, sat_ids[i], fontsize=9, ha='right')

    edge_list = data.edge_index.t().tolist()
    distances = data.edge_attr.tolist()
    for (i, j), d in zip(edge_list, distances):
        x_vals = [positions[i][0], positions[j][0]]
        y_vals = [positions[i][1], positions[j][1]]
        z_vals = [positions[i][2], positions[j][2]]
        ax.plot(x_vals, y_vals, z_vals, color='gray', linewidth=1)
        mid_x = (x_vals[0] + x_vals[1]) / 2
        mid_y = (y_vals[0] + y_vals[1]) / 2
        mid_z = (z_vals[0] + z_vals[1]) / 2
        ax.text(mid_x, mid_y, mid_z, f"{d:.1f} km", color='red', fontsize=8)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km")
    ax.set_zlabel("Z (km")
    ax.set_title("3D Satellite Interaction Graph")
    plt.legend()
    plt.tight_layout()
    plt.show()
