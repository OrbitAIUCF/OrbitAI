    # Interactive 3D Satellite Graph Animation with Plotly
    # Simple 2D & 3D Graphs Using Matplotlib
    # Purpose: Sequentially visualize satellite interactions over time using a slider

import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
import random
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# -------------------------------
# Graph Animation Visualization (3D)
# -------------------------------
def animate_graph_3d_interactive(graph_list):
    all_frames = []

     # 1) Compute global XYZ ranges
    all_positions = np.vstack([g.x[:, :3].numpy() for g in graph_list])
    buffer = 500  # Extra space to keep axis ranges consistent visually
    x_min, x_max = np.min(all_positions[:, 0]), np.max(all_positions[:, 0])
    y_min, y_max = np.min(all_positions[:, 1]), np.max(all_positions[:, 1])
    z_min, z_max = np.min(all_positions[:, 2]), np.max(all_positions[:, 2])
    x_range = [x_min - buffer, x_max + buffer]
    y_range = [y_min - buffer, y_max + buffer]
    z_range = [z_min - buffer, z_max + buffer]

    cmap = plt.get_cmap("RdYlGn_r")

    for graph in graph_list:
        positions = graph.x[:, :3].numpy()
        directed_edges = graph.edge_index.t().tolist()
        sat_ids = getattr(graph, 'sat_ids', [str(i) for i in range(len(positions))])
        timestamp = getattr(graph, 'timestamp', "T")

        # 2) Pull out & normalize attention
        attn_weights = getattr(graph, 'attn_weights', None)
        if attn_weights is not None:
            # 1) pull out the raw attention array [num_edges, heads]
            attn_array = attn_weights.detach().cpu().numpy()
            # 2) collapse across heads into a single scalar per edge
            if attn_array.ndim > 1:
                attn_array = attn_array.mean(axis=1)
            # 3) normalize to [0,1]
            norm = (attn_array - attn_array.min()) / (attn_array.max() - attn_array.min() + 1e-8)
        else:
            edge_colors = ["gray"] * len(edge_index)
            norm = np.zeros(len(directed_edges), dtype=float)

        # 3) Collapse bidirectional edges into one entry, picking the direction
        #    with the higher normalized weight.
        edge_map = {}
        for (i,j), w in zip(directed_edges, norm.tolist()):
            key = frozenset((i,j))
            edge_map.setdefault(key, []).append(((i,j), w))

        # 4) Build a list of unique edges with dist, rel_vel, sorted by weight
        edges_info = []
        for key, entries in edge_map.items():
            # pick the (i,j) tuple whose w is highest
            (i,j), weight = max(entries, key=lambda x: x[1])

            pos_i, pos_j = positions[i], positions[j]
            dist    = np.linalg.norm(pos_i - pos_j)
            vel_i   = graph.x[i,3:6].cpu().numpy()
            vel_j   = graph.x[j,3:6].cpu().numpy()
            rel_vel = np.linalg.norm(vel_i - vel_j)

            # map weight→RGBA via plasma
            r,g,b,_ = cmap(float(weight))
            color = f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1)"

            edges_info.append({
                'i': i, 'j': j,
                'dist': dist, 'rel_vel': rel_vel,
                'weight': weight, 'color': color
            })

        # sort so highest-attention edges come first
        edges_info.sort(key=lambda e: e['weight'], reverse=True)

        # 5) Build line segments + table rows
        edge_lines = []
        edge_color_list = []
        table_rows = []  # [source, dist, rel_vel, severity, target, font_color]

        for info in edges_info:
            i, j = info['i'], info['j']
            c = info['color']
            edge_lines.extend([positions[i], positions[j], [None,None,None]])
            edge_color_list.extend([c, c, 'rgba(0,0,0,0)'])

            # severity bucket
            w = info['weight']
            if w > 0.66: sev = "HIGH"
            elif w > 0.33: sev = "MEDIUM"
            else: sev = "LOW"

            table_rows.append([
                sat_ids[i],
                f"{info['dist']:.1f} km",
                f"{info['rel_vel']:.1f} m/s",
                sev,
                sat_ids[j],
                c   # font color for this row’s “Attention” cell
            ])

        # 6) Scatter3d trace for edges & nodes
        edge_trace = go.Scatter3d(
            x=[p[0] for p in edge_lines],
            y=[p[1] for p in edge_lines],
            z=[p[2] for p in edge_lines],
            mode='lines',
            line=dict(color=edge_color_list, width=3),
            showlegend=False
        )
        node_trace = go.Scatter3d(
            x=positions[:,0], y=positions[:,1], z=positions[:,2],
            mode='markers+text',
            text=sat_ids, textposition='top center',
            marker=dict(size=5, color='skyblue'),
            name=str(timestamp)
        )

        # 7) Table with colored “Attention” text & full vertical domain
        cols = list(zip(*table_rows))
        table_trace = go.Table(
        # 1) make it float over the top 25% of the figure
        domain=dict(x=[0.0, 1.0], y=[0.75, 1.0]),

        # 2) yellow header with black text
        header=dict(
            values=["Source","Distance","Relative Velocity","Attention","Target"],
            fill_color="yellow",
            font_color="black",
            align="left"
        ),

        # 3) black cell background, white text everywhere except the
        #    “Attention” column, which still gets its per-row RGBA color
        cells=dict(
            values=cols[:-1],      # all columns except our font-color list
            fill_color="black",
            align="left",
            font_color=[
                ["white"] * len(cols[0]),   # Source
                ["white"] * len(cols[0]),   # Distance
                ["white"] * len(cols[0]),   # Rel. Velocity
                cols[-1],                   # Attention (our RGBA list)
                ["white"] * len(cols[0])    # Target
            ]
        )
    )

        all_frames.append(go.Frame(
            data=[edge_trace, node_trace, table_trace],
            name=str(timestamp)
        ))

    # Define default animation frame duration
    default_duration = 250  # ms

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
        args=[None, {
            "mode": "immediate",
            "frame": {"duration": 1250, "redraw": True},
            "transition": {"duration": 0}
        }]
    )

    # Faster/Slower Buttons Using relayout (updates frame duration)
    faster_button = dict(
        label="Faster",
        method="animate",
        args=[None, {
            "frame": {"duration": default_duration-100, "redraw": True},
            "fromcurrent": True,
            "transition": {"duration": 0}
        }]
    )
    slower_button = dict(
        label = "Slower",
        method="animate",
        args=[None, {
            "frame": {"duration": default_duration+200, "redraw": True},
            "fromcurrent": True,
            "transition": {"duration": 0}
        }]
    )

    fig = go.Figure(
        data=all_frames[0].data,
        frames=all_frames,
        layout=go.Layout(
            title="OrbitAI Satellite Interaction Timeline",
            scene=dict(
                domain=dict(x=[0,1],y=[0,0.7]),
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
                    x=0.0, y=1  # adjust position if needed
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
