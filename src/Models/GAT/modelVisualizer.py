    # Interactive 3D Satellite Graph Animation with Plotly
    # Simple 2D & 3D Graphs Using Matplotlib
    # Purpose: Sequentially visualize satellite interactions over time using a slider

import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import random
import plotly.graph_objects as go
import numpy as np

# -------------------------------
# Graph Animation Visualization (3D)
# -------------------------------
def animate_graph_3d_interactive(graph_list):
    all_frames = []

    # Establish global axis ranges based on all positions
    all_positions = np.vstack([g.x[:, :3].numpy() for g in graph_list])
    buffer = 500  # Extra space to keep axis ranges consistent visually
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])
    x_range = [x_min - buffer, x_max + buffer]
    y_range = [y_min - buffer, y_max + buffer]
    z_range = [z_min - buffer, z_max + buffer]

    for graph in graph_list:
        positions = graph.x[:, :3].numpy()
        edge_index = graph.edge_index.t().tolist()
        sat_ids = graph.sat_ids if hasattr(graph, 'sat_ids') else [str(i) for i in range(len(positions))]
        timestamp = graph.timestamp if hasattr(graph, 'timestamp') else "T"

        attn_weights = getattr(graph, 'attn_weights', None)
        edge_colors = []
        if attn_weights is not None:
            attn_array = attn_weights.detach().cpu().numpy()
            norm = (attn_array - attn_array.min()) / (attn_array.max() - attn_array.min() + 1e-8)
            cmap = plt.get_cmap("plasma")
            edge_colors = [f"rgba{tuple(int(c*255) for c in cmap(w)[:3]) + (1,)}" for w in norm]
        else:
            edge_colors = ["gray"] * len(edge_index)

        edge_lines = []
        edge_color_list = []
        for k, (i, j) in enumerate(edge_index):
            edge_lines.extend([positions[i], positions[j], [None, None, None]])
            edge_color_list.extend([edge_colors[k], None])

        edge_trace = go.Scatter3d(
            x=[pt[0] for pt in edge_lines],
            y=[pt[1] for pt in edge_lines],
            z=[pt[2] for pt in edge_lines],
            mode='lines',
            line=dict(color=edge_color_list, width=3),
            showlegend=False
        )

        node_trace = go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers+text',
            text=sat_ids,
            textposition="top center",
            marker=dict(size=5, color='skyblue'),
            name=str(timestamp)
        )

        frame = go.Frame(data=[edge_trace, node_trace], name=str(timestamp))
        all_frames.append(frame)

    # Define default animation frame duration
    default_duration = 350  # ms

    # Standard Play/Pause and Step Buttons
    play_button = dict(
        label="Play",
        method="animate",
        args=[None, {
            "frame": {"duration": default_duration, "redraw": True},
            "fromcurrent": True,
            "transition": {"duration": 0}
        }]
    )
    pause_button = dict(
        label="Pause",
        method="animate",
        args=[[None], {
            "frame": {"duration": 0, "redraw": False},
            "mode": "immediate"
        }]
    )
    step_forward_button = dict(
        label="Step Forward",
        method="animate",
        args=[[None], {
            "mode": "immediate",
            "frame": {"duration": 0, "redraw": True},
            "transition": {"duration": 0}
        }]
    )

    # Faster/Slower Buttons Using relayout (updates frame duration)
    faster_button = dict(
        label="Faster",
        method="relayout",
        args=[{"frame.duration": 100}]
    )
    slower_button = dict(
        label="Slower",
        method="relayout",
        args=[{"frame.duration": 400}]
    )

    fig = go.Figure(
        data=all_frames[0].data,
        frames=all_frames,
        layout=go.Layout(
            title="OrbitAI Satellite Interaction Timeline",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
                xaxis=dict(range=x_range),
                yaxis=dict(range=y_range),
                zaxis=dict(range=z_range),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1)
            ),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        play_button, pause_button, step_forward_button,  # step_backward not natively supported
                        faster_button, slower_button
                    ],
                    x=0.0, y=1.15  # adjust position if needed
                )
            ],
            sliders=[dict(
                steps=[
                    dict(method='animate',
                        args=[[f.name],
                            dict(mode='immediate',
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0)
                            )
                            ],
                        label=f.name)
                    for f in all_frames
                ],
                transition=dict(duration=0),
                x=0, y=0,
                currentvalue=dict(font=dict(size=14), prefix="Time: ", visible=True),
                len=1.0
            )]
        )
    )

    fig.show()



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
