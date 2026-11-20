import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist


def plot_autoencoder_architecture(layer_dims: list[int], activation: str, tr) -> go.Figure:
    """Draw a clean interactive neural network diagram of Encoder -> Latent Space -> Decoder."""
    fig = go.Figure()
    x_positions = [0.0, 1.5, 3.0, 4.5, 6.0]
    colors = ["#4C78A8", "#00CC96", "#F59E0B", "#AB63FA", "#4C78A8"]
    layer_titles = [
        tr(f"Input x<br>({layer_dims[0]}D)", f"Entrada x<br>({layer_dims[0]}D)"),
        tr(f"Encoder<br>({layer_dims[1]} units)", f"Codificador<br>({layer_dims[1]} neuronas)"),
        tr(f"Latent Space z<br>({layer_dims[2]}D)", f"Espacio Latente z<br>({layer_dims[2]}D)"),
        tr(f"Decoder<br>({layer_dims[3]} units)", f"Decodificador<br>({layer_dims[3]} neuronas)"),
        tr(f"Output x̂<br>({layer_dims[4]}D)", f"Salida x̂<br>({layer_dims[4]}D)"),
    ]

    max_vis = 8
    node_coords: list[list[tuple[float, float]]] = []
    for x_pos, dim in zip(x_positions, layer_dims):
        n_vis = min(dim, max_vis)
        ys = np.linspace(-n_vis / 2.0, n_vis / 2.0, n_vis) if n_vis > 1 else np.array([0.0])
        node_coords.append([(x_pos, float(y)) for y in ys])

    # Connections between adjacent layers
    edge_x, edge_y = [], []
    for i in range(len(node_coords) - 1):
        for x0, y0 in node_coords[i]:
            for x1, y1 in node_coords[i + 1]:
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(150, 160, 175, 0.35)", width=1.1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Highlight bottleneck region
    fig.add_vrect(
        x0=2.55,
        x1=3.45,
        fillcolor="rgba(245, 158, 11, 0.12)",
        line=dict(color="#F59E0B", width=1.5, dash="dash"),
        layer="below",
    )

    # Draw nodes per layer
    for i, (coords, color, title, dim) in enumerate(zip(node_coords, colors, layer_titles, layer_dims)):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        size = 24 if i == 2 else 18
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(size=size, color=color, line=dict(width=1.8, color="#1F2937")),
                name=title.replace("<br>", " "),
                hovertemplate=f"{title.replace('<br>', ' ')}<extra></extra>",
                showlegend=False,
            )
        )
        # Layer label at bottom
        fig.add_annotation(
            x=x_positions[i],
            y=-5.1,
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=12, color="#1F2937"),
        )
        if dim > max_vis:
            fig.add_annotation(
                x=x_positions[i],
                y=4.45,
                text=tr(f"(showing {max_vis} of {dim})", f"(mostrando {max_vis} de {dim})"),
                showarrow=False,
                font=dict(size=10, color="#6B7280"),
            )

    # Top stage headers
    fig.add_annotation(
        x=0.75,
        y=5.3,
        text=tr(f"<b>ENCODER f(x)</b> ({activation})", f"<b>CODIFICADOR f(x)</b> ({activation})"),
        showarrow=False,
        font=dict(size=13, color="#009E73"),
    )
    fig.add_annotation(
        x=3.0,
        y=5.3,
        text=tr("<b>BOTTLENECK z</b>", "<b>CUELLO DE BOTELLA z</b>"),
        showarrow=False,
        font=dict(size=13, color="#D97706"),
    )
    fig.add_annotation(
        x=5.25,
        y=5.3,
        text=tr(f"<b>DECODER g(z)</b> ({activation})", f"<b>DECODIFICADOR g(z)</b> ({activation})"),
        showarrow=False,
        font=dict(size=13, color="#7C3AED"),
    )

    fig.update_layout(
        height=360,
        margin=dict(l=15, r=15, t=25, b=15),
        xaxis=dict(visible=False, range=[-0.6, 6.6]),
        yaxis=dict(visible=False, range=[-6.0, 6.0]),
        template="plotly_white",
    )
    return fig


def plot_condensed_tree(
    condensed: dict[int, dict],
    eom_selected: set[int],
    leaf_selected: set[int],
    highlight_mode: str,
    tr,
) -> go.Figure:
    """
    Plot the HDBSCAN Condensed Tree where:
      - Vertical axis (top to bottom): lambda = 1 / distance (density level)
      - Width of each shaded funnel: number of points remaining in that cluster
      - Shaded area of each cluster = Cluster Stability S(C)
      - Highlighted borders indicate EOM or Leaf selected clusters
    """
    fig = go.Figure()
    if not condensed:
        return fig

    # Assign x-positions via post-order traversal (leaves evenly spaced)
    x_pos: dict[int, float] = {}
    leaf_counter = 0

    def assign_x(cid: int) -> float:
        nonlocal leaf_counter
        children = condensed[cid]["children"]
        if not children:
            x_pos[cid] = float(leaf_counter)
            leaf_counter += 1
            return x_pos[cid]
        xs = [assign_x(ch) for ch in children]
        x_pos[cid] = float(np.mean(xs))
        return x_pos[cid]

    assign_x(0)

    total_n = max(1, condensed[0]["initial_size"])
    max_half_w = 0.36
    palette = px.colors.qualitative.Plotly

    selected_set = eom_selected if highlight_mode == "eom" else leaf_selected
    max_lam = max((info["lambda_death"] for info in condensed.values()), default=1.0)
    # Clamp extreme lambda spikes for clean visual scaling (99th percentile cap)
    finite_lams = [info["lambda_death"] for info in condensed.values() if info["lambda_death"] < 1e4]
    lam_cap = max(finite_lams) * 1.05 if finite_lams else max_lam

    # Draw connector lines between parents and children first
    for cid, info in condensed.items():
        if info["children"]:
            lam_split = min(info["lambda_death"], lam_cap)
            c1, c2 = info["children"]
            fig.add_trace(
                go.Scatter(
                    x=[x_pos[c1], x_pos[c2]],
                    y=[lam_split, lam_split],
                    mode="lines",
                    line=dict(color="#6B7280", width=2),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Draw shaded stability funnels for each cluster
    sel_color_idx = 0
    for cid in sorted(condensed.keys()):
        info = condensed[cid]
        cx = x_pos[cid]
        events = info["events"]
        lams = [min(ev[0], lam_cap) for ev in events]
        sizes = [ev[1] for ev in events]
        if len(lams) == 1:
            lams.append(min(info["lambda_death"], lam_cap))
            sizes.append(sizes[0])

        # Build stepped polygon (left edge going down, right edge going up)
        left_x, left_y = [], []
        right_x, right_y = [], []
        for idx_ev in range(len(lams)):
            w = max(0.03, (sizes[idx_ev] / total_n) * max_half_w)
            y_val = lams[idx_ev]
            if idx_ev > 0:
                prev_w = max(0.03, (sizes[idx_ev - 1] / total_n) * max_half_w)
                left_x.append(cx - prev_w)
                left_y.append(y_val)
                right_x.append(cx + prev_w)
                right_y.append(y_val)
            left_x.append(cx - w)
            left_y.append(y_val)
            right_x.append(cx + w)
            right_y.append(y_val)

        poly_x = left_x + right_x[::-1] + [left_x[0]]
        poly_y = left_y + right_y[::-1] + [left_y[0]]

        is_sel = cid in selected_set
        if is_sel:
            base_col = palette[sel_color_idx % len(palette)]
            sel_color_idx += 1
            fill_col = base_col
            line_col = "#111827"
            line_w = 2.4
            opacity = 0.78
        elif cid == 0:
            fill_col = "rgba(156, 163, 175, 0.35)"
            line_col = "#6B7280"
            line_w = 1.2
            opacity = 0.55
        else:
            fill_col = "rgba(180, 190, 205, 0.42)"
            line_col = "#9CA3AF"
            line_w = 1.2
            opacity = 0.6

        status_txt = (
            tr("SELECTED", "SELECCIONADO")
            if is_sel
            else (tr("Root", "Raíz") if cid == 0 else tr("Not selected", "No seleccionado"))
        )
        hover = (
            f"<b>Node C{cid}</b> ({status_txt})<br>"
            f"{tr('Initial points', 'Puntos iniciales')}: {info['initial_size']}<br>"
            f"λ_birth: {info['lambda_birth']:.3f} → λ_death: {info['lambda_death']:.3f}<br>"
            f"<b>{tr('Stability S(C)', 'Estabilidad S(C)')}: {info['stability']:.2f}</b><br>"
            f"{tr('Leaf?', '¿Hoja?')}: {'Yes' if info['is_leaf'] else 'No'}"
        )

        fig.add_trace(
            go.Scatter(
                x=poly_x,
                y=poly_y,
                fill="toself",
                fillcolor=fill_col,
                opacity=opacity,
                mode="lines",
                line=dict(color=line_col, width=line_w),
                name=f"C{cid} (S={info['stability']:.1f})",
                text=hover,
                hoverinfo="text",
                showlegend=is_sel,
            )
        )

        # Label inside/near selected clusters
        if is_sel or cid == 0:
            mid_y = 0.5 * (min(info["lambda_birth"], lam_cap) + min(info["lambda_death"], lam_cap))
            lbl_text = f"<b>C{cid}</b><br>S={info['stability']:.1f}" if cid != 0 else tr("Root", "Raíz")
            fig.add_annotation(
                x=cx,
                y=mid_y,
                text=lbl_text,
                showarrow=False,
                font=dict(size=10, color="#111827"),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor=line_col,
                borderwidth=1,
            )

    mode_label = "EOM (Excess of Mass)" if highlight_mode == "eom" else "Leaf"
    fig.update_layout(
        title=tr(
            f"Condensed Cluster Tree — Highlighting {mode_label} Selection (Area = Stability)",
            f"Árbol Condensado de Clusters — Resaltando Selección {mode_label} (Área = Estabilidad)",
        ),
        xaxis=dict(visible=False),
        yaxis=dict(
            title=tr("Density Level λ = 1 / distance (increases downward ↓)", "Nivel de Densidad λ = 1 / distancia (aumenta hacia abajo ↓)"),
            autorange="reversed",
        ),
        height=470,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.16),
    )
    return fig


def plot_mst_graph(
    X: np.ndarray,
    mst_edges: list[tuple[int, int, float]],
    threshold: float,
    labels: np.ndarray,
    tr,
) -> go.Figure:
    """Plot the Minimum Spanning Tree (MST) with kept edges vs. cut edges above threshold."""
    fig = go.Figure()

    kept_x, kept_y = [], []
    cut_x, cut_y = [], []
    for u, v, w in mst_edges:
        if w <= threshold:
            kept_x.extend([X[u, 0], X[v, 0], None])
            kept_y.extend([X[u, 1], X[v, 1], None])
        else:
            cut_x.extend([X[u, 0], X[v, 0], None])
            cut_y.extend([X[u, 1], X[v, 1], None])

    if kept_x:
        fig.add_trace(
            go.Scatter(
                x=kept_x,
                y=kept_y,
                mode="lines",
                name=tr(f"Active MST edges (w ≤ {threshold:.2f})", f"Aristas MST activas (w ≤ {threshold:.2f})"),
                line=dict(color="rgba(55, 65, 81, 0.55)", width=1.8),
                hoverinfo="skip",
            )
        )
    if cut_x:
        fig.add_trace(
            go.Scatter(
                x=cut_x,
                y=cut_y,
                mode="lines",
                name=tr(f"Cut MST edges (w > {threshold:.2f})", f"Aristas MST cortadas (w > {threshold:.2f})"),
                line=dict(color="rgba(239, 68, 68, 0.55)", width=1.4, dash="dot"),
                hoverinfo="skip",
            )
        )

    palette = px.colors.qualitative.Plotly
    unique_lbls = sorted(set(labels))
    for lbl in unique_lbls:
        mask = labels == lbl
        if lbl == -1:
            name = tr("Noise / Small (< min_cluster_size)", "Ruido / Pequeño (< min_cluster_size)")
            col = "#9CA3AF"
            sym = "x"
            sz = 6
        else:
            name = f"Cluster {lbl + 1}"
            col = palette[lbl % len(palette)]
            sym = "circle"
            sz = 7
        fig.add_trace(
            go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode="markers",
                name=name,
                marker=dict(size=sz, color=col, symbol=sym, line=dict(width=0.5, color="white")),
            )
        )

    fig.update_layout(
        title=tr(
            f"Minimum Spanning Tree (MST) Cut at Mutual Reachability ≤ {threshold:.2f}",
            f"Árbol de Expansión Mínima (MST) Cortado en Alcanzabilidad Mutua ≤ {threshold:.2f}",
        ),
        xaxis_title="$x_1$",
        yaxis_title="$x_2$",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=490,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


def plot_centers_comparison(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    medoids: np.ndarray,
    title: str,
    tr,
) -> tuple[go.Figure, list[dict]]:
    """Plot clusters alongside HDBSCAN Centroids vs. Medoids and measure empty-space gaps."""
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly

    for lbl in sorted(set(labels)):
        mask = labels == lbl
        if lbl == -1:
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    name=tr("Noise (-1)", "Ruido (-1)"),
                    marker=dict(size=6, symbol="x", color="#9CA3AF"),
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=X[mask, 0],
                    y=X[mask, 1],
                    mode="markers",
                    name=f"Cluster {lbl + 1}",
                    marker=dict(size=7, color=palette[lbl % len(palette)], opacity=0.72),
                )
            )

    rows = []
    n_clusters = len(centroids) if centroids is not None else 0
    for c in range(n_clusters):
        c_pts = X[labels == c]
        if len(c_pts) == 0:
            continue
        cent = centroids[c]
        med = medoids[c]
        d_cent = cdist([cent], c_pts)[0]
        nearest_idx = int(np.argmin(d_cent))
        gap_cent = float(d_cent[nearest_idx])
        nearest_pt = c_pts[nearest_idx]

        # Dashed connector from centroid to nearest real cluster point
        fig.add_trace(
            go.Scatter(
                x=[cent[0], nearest_pt[0]],
                y=[cent[1], nearest_pt[1]],
                mode="lines",
                line=dict(color="#DC2626", width=2, dash="dash"),
                name=tr("Centroid gap to real data", "Brecha del centroide a datos reales") if c == 0 else None,
                showlegend=(c == 0),
                hoverinfo="skip",
            )
        )

        rows.append(
            {
                "Cluster": f"Cluster {c + 1}",
                "Centroid (x1, x2)": f"({cent[0]:.2f}, {cent[1]:.2f})",
                "Medoid (x1, x2)": f"({med[0]:.2f}, {med[1]:.2f})",
                "Centroid Gap": round(gap_cent, 3),
                "Medoid Gap": 0.0,
            }
        )

    if n_clusters > 0:
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode="markers",
                name=tr("Centroid (weighted mean — can lie in empty space!)", "Centroide (media ponderada — ¡puede caer en el vacío!)"),
                marker=dict(size=18, symbol="star", color="#EF4444", line=dict(width=1.5, color="#111827")),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=medoids[:, 0],
                y=medoids[:, 1],
                mode="markers",
                name=tr("Medoid (actual data point inside cluster)", "Medoide (punto real dentro del cluster)"),
                marker=dict(size=15, symbol="diamond", color="#10B981", line=dict(width=2, color="#111827")),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="$x_1$",
        yaxis_title="$x_2$",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig, rows
