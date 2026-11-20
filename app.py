import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from models import (
    NumpyAutoencoder,
    compute_hdbscan_steps,
    load_real_highd_dataset,
    make_3d_manifold,
    make_nonconvex_centers_dataset,
    make_varying_density_dataset,
    mst_connected_components,
)
from viz import (
    plot_autoencoder_architecture,
    plot_centers_comparison,
    plot_condensed_tree,
    plot_mst_graph,
)

# --- Configuration ---
st.set_page_config(
    page_title="Autoencoders & HDBSCAN: Theory, Math & Interactive Visualizations",
    layout="wide",
)

# --- Sidebar: Author Info ---
with st.sidebar:
    st.header("Author")
    st.markdown("Daniel Edgardo Rodríguez Rivera")
    st.markdown("El Salvador")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/daniel-rodriguez-sv/")
    with col2:
        st.link_button("GitHub", "https://github.com/danielrodriguezrivera/")
    st.markdown("---")

# --- Language Selector ---
lang_choice = st.sidebar.radio(
    "Language / Idioma:",
    ["Español", "English"],
    horizontal=True,
)
ES = lang_choice == "Español"
st.sidebar.markdown("---")


def tr(en_text: str, es_text: str) -> str:
    """Return Spanish or English text based on selected language."""
    return es_text if ES else en_text


def format_cluster_labels(labels: np.ndarray) -> list[str]:
    out = []
    for c in labels:
        if c == -1:
            out.append(tr("Noise (-1)", "Ruido (-1)"))
        else:
            out.append(f"Cluster {c + 1}")
    return out


# --- Cached Wrappers ---
@st.cache_data
def train_cached_autoencoder(
    X_in: np.ndarray,
    X_target: np.ndarray | None,
    hidden_dim: int,
    latent_dim: int,
    activation: str,
    epochs: int,
    lr: float = 0.015,
    batch_size: int | None = 64,
    seed: int = 42,
):
    ae = NumpyAutoencoder(
        input_dim=X_in.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        activation=activation,
        lr=lr,
        seed=seed,
    )
    ae.fit(X_in, X_target=X_target, epochs=epochs, batch_size=batch_size)
    return ae


@st.cache_data
def cached_hdbscan_steps(X: np.ndarray, min_cluster_size: int, min_samples: int):
    return compute_hdbscan_steps(X, min_cluster_size=min_cluster_size, min_samples=min_samples)


# --- Header ---
st.title(
    tr(
        "Autoencoders & HDBSCAN: Essential Theory, Real-World Cases & Interactive Visualizations",
        "Autoencoders y HDBSCAN: Teoría Esencial, Casos Reales y Visualizaciones Interactivas",
    )
)
st.markdown(
    tr(
        "Explore how **Autoencoders** compress high-dimensional data into nonlinear **latent spaces** and how **HDBSCAN** discovers variable-density clusters via **Minimum Spanning Trees (MST)** and **Cluster Stability**. Use the sidebar to navigate through the interactive modules.",
        "Explora cómo los **Autoencoders** comprimen datos de alta dimensión en **espacios latentes** no lineales y cómo **HDBSCAN** descubre grupos de densidad variable mediante **Árboles de Expansión Mínima (MST)** y **Estabilidad de Clusters**. Usa la barra lateral para navegar por los módulos interactivos.",
    )
)

# --- Master Theory Expander ---
with st.expander(
    tr(
        "Read the Essential Theory, Math & Analogies (Autoencoders & HDBSCAN)",
        "Leer la Teoría Esencial, Matemática y Analogías (Autoencoders y HDBSCAN)",
    ),
    expanded=False,
):
    if not ES:
        st.markdown(
            r"""
            ### 1. Autoencoders: The "Sketch Artist & Sculptor" Analogy
            Imagine a game between two artists separated by a wall, with only a tiny **sticky note** that fits **just 2 numbers** passed through a slot:
            - The **Sketch Artist (Encoder)** looks at a complex object (e.g., an 8×8 handwritten digit made of 64 pixels) and must summarize its *most essential traits* into those 2 numbers on the sticky note (**Latent Space**).
            - The **Sculptor (Decoder)** on the other side has never seen the original object — they receive *only* the sticky note and must rebuild the 64-pixel image from scratch.
            - Finally, a judge compares the sculpture against the original (**Reconstruction Loss**). Because the sticky note is so small (the **bottleneck**), the duo **cannot waste space memorizing random noise** — they are forced to invent a smart, compressed coordinate system for the true structure of the data!

            Formally, an **Autoencoder** is a self-supervised neural network trained to reconstruct its own input ($\hat{x} \approx x$) through that bottleneck:
            1. **Encoder $f_\phi(x)$:** Compresses high-dimensional input $x \in \mathbb{R}^{D}$ into a compact latent code $z \in \mathbb{R}^{d}$ ($d \ll D$):
               $$z = f_\phi(x)$$
            2. **Latent Space ($z$):** The compressed representation (the sticky note / bottleneck) that captures the essential manifold of the data.
            3. **Decoder $g_\theta(z)$:** Reconstructs the original data from the latent coordinates:
               $$\hat{x} = g_\theta(z) = g_\theta(f_\phi(x))$$
            4. **Reconstruction Loss (MSE):** Measures how different the sculptor's reconstruction is from the original input:
               $$\mathcal{L}(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2$$
            - **Dimensionality Reduction vs. PCA:** If the Encoder and Decoder use only **linear** activations, an Autoencoder spans the exact same subspace as **PCA**. With **nonlinear activations** (`tanh`, `ReLU`), it bends to unroll curved manifolds (Swiss rolls, images, complex sensor signals) that PCA flattens poorly.

            ---

            ### 2. HDBSCAN: The "Archipelago Rising from the Sea" Analogy
            Think of your data density as an **underwater mountain range** (tall peaks = dense clusters, shallow hills = sparse clusters, flat ocean floor = noise).
            - **DBSCAN** picks **one single sea level** ($\varepsilon$): if the sea is too high, shorter sparse islands drown and become noise; if the sea is too low, nearby tall islands merge into one giant landmass.
            - **HDBSCAN** (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*, Campello, Moulavi & Sander, 2013) instead **raises the sea level continuously** (increasing density threshold $\lambda = 1/d$) and watches the entire history of the archipelago!
              - As the water rises, a continent splits into islands, and islands split into smaller peaks.
              - Tiny rocks with fewer than `min_cluster_size` points simply **erode into the ocean (noise)** instead of counting as real islands.
              - Islands that **stay above water across a wide range of sea levels** with many points have high **Cluster Stability** — HDBSCAN keeps the most persistent islands, even if one island is a tall dense spire and another is a broad low plateau!

            Formally, HDBSCAN executes this in 5 steps:
            1. **Core Distance & Mutual Reachability Distance:**
               For $k = \text{min\_samples}$, let $\text{core}_k(a)$ be the distance from $a$ to its $k$-th nearest neighbor. Define:
               $$d_{\text{mreach-}k}(a, b) = \max\{\text{core}_k(a),\; \text{core}_k(b),\; d(a, b)\}$$
               Dense points stay close together, while sparse outliers are automatically pushed far away!
            2. **Minimum Spanning Tree (MST):** Build the cheapest tree connecting all $N$ points using $d_{\text{mreach-}k}$ as edge weights. Cutting the heaviest MST edges simulates raising the sea level from coarse to fine.
            3. **Condensed Tree (`min_cluster_size`):** Walk down the hierarchy. When a split produces a branch smaller than `min_cluster_size`, those points simply **fall out as noise** instead of creating tiny spurious clusters.
            4. **Cluster Stability:** Define density level $\lambda = \frac{1}{d_{\text{mreach}}}$. The **stability** of a cluster $C$ measures how long its points persist across density levels (visually, its **shaded area** in the condensed tree):
               $$S(C) = \sum_{p \in C} \left(\lambda_p - \lambda_{\text{birth}}(C)\right)$$
            5. **`cluster_selection_method` (`eom` vs. `leaf`):**
               - **`eom` (Excess of Mass):** Maximizes total stability bottom-up: if $S(C) \ge \hat{S}(C_{\text{left}}) + \hat{S}(C_{\text{right}})$, keep the whole island (parent $C$); otherwise keep its separate peaks (children). Produces **fewer, larger, most persistent** clusters.
               - **`leaf`:** Always picks the bottom **leaf clusters** (individual mountain peaks) of the condensed tree. Produces **more, smaller, finer-grained** homogeneous clusters.
            6. **`store_centers` (`centroid` vs. `medoid`):**
               - **`centroid`:** Probability-weighted average position of a cluster — fast, but **can land in empty space** (in the water!) on curved/non-convex shapes (rings, moons).
               - **`medoid`:** An **actual data point** in the cluster minimizing total distance to all other members — **guaranteed to stand on dry land (the real data manifold)**.
            """
        )
    else:
        st.markdown(
            r"""
            ### 1. Autoencoders: La Analogía del "Caricaturista y el Escultor"
            Imagina un juego entre dos artistas separados por una pared, con solo una **tarjeta diminuta** donde caben **apenas 2 números** para comunicarse:
            - El **Caricaturista (Codificador / *Encoder*)** observa un objeto complejo (por ejemplo, un dígito manuscrito de 8×8 = 64 píxeles) y debe resumir sus *rasgos más esenciales* en esos 2 números de la tarjeta (**Espacio Latente**).
            - El **Escultor (Decodificador / *Decoder*)** al otro lado jamás vio el objeto original: recibe *únicamente* esa tarjeta con 2 números y debe reconstruir la imagen completa de 64 píxeles desde cero.
            - Al final, un juez compara la escultura con el modelo original (**Pérdida de Reconstrucción**). Como la tarjeta es tan estrecha (el **cuello de botella**), ambos **no pueden desperdiciar espacio memorizando ruido**: se ven obligados a inventar un sistema de coordenadas inteligente que capture solo la esencia de los datos.

            Formalmente, un **Autoencoder** es una red neuronal autosupervisada entrenada para reconstruir su propia entrada ($\hat{x} \approx x$) pasando por ese cuello de botella:
            1. **Codificador (*Encoder*) $f_\phi(x)$:** Comprime la entrada de alta dimensión $x \in \mathbb{R}^{D}$ en un código latente compacto $z \in \mathbb{R}^{d}$ ($d \ll D$):
               $$z = f_\phi(x)$$
            2. **Espacio Latente ($z$):** La representación comprimida (la tarjeta / cuello de botella) que captura la estructura esencial (*manifold*) de los datos.
            3. **Decodificador (*Decoder*) $g_\theta(z)$:** Reconstruye el dato original a partir de las coordenadas latentes:
               $$\hat{x} = g_\theta(z) = g_\theta(f_\phi(x))$$
            4. **Pérdida de Reconstrucción (MSE):** Mide qué tan diferente es la reconstrucción del escultor frente a la entrada original:
               $$\mathcal{L}(\phi, \theta) = \frac{1}{N} \sum_{i=1}^{N} \|x_i - \hat{x}_i\|_2^2$$
            - **Reducción de Dimensionalidad vs. PCA:** Si el Codificador y Decodificador usan solo activaciones **lineales**, el Autoencoder recupera el mismo subespacio que **PCA**. Con **activaciones no lineales** (`tanh`, `ReLU`), logra doblarse y desenrollar geometrías curvas (rollos suizos, imágenes, señales complejas) donde PCA pierde información.

            ---

            ### 2. HDBSCAN: La Analogía del "Archipiélago que Emerge del Mar"
            Imagina la densidad de tus datos como una **cordillera sumergida bajo el océano** (picos altos = clusters muy densos, mesetas bajas = clusters amplios y dispersos, fondo marino plano = ruido):
            - **DBSCAN** elige **un único nivel fijo del mar** ($\varepsilon$): si el agua está demasiado alta, las islas bajas y dispersas se hunden y se vuelven ruido; si el agua está demasiado baja, las islas altas cercanas quedan pegadas como un solo continente.
            - **HDBSCAN** (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*, Campello, Moulavi y Sander, 2013) en cambio **sube el nivel del mar de forma continua** (aumentando el nivel de densidad $\lambda = 1/d$) y observa toda la historia del archipiélago:
              - Al subir el agua, un continente se divide en islas y las islas se separan en picos individuales.
              - Los islotes diminutos con menos de `min_cluster_size` puntos simplemente **se desprenden y desaparecen en el mar (ruido)** en lugar de contarse como islas reales.
              - Las islas que **resisten sobre el agua durante un gran rango de niveles** con muchos puntos tienen alta **Estabilidad**: así HDBSCAN conserva las islas más persistentes, sin importar si una es un pico alto y denso y otra es una meseta amplia y dispersa.

            Formalmente, HDBSCAN ejecuta este proceso en 5 pasos:
            1. **Distancia Núcleo y Distancia de Alcanzabilidad Mutua (*Mutual Reachability*):**
               Para $k = \text{min\_samples}$, sea $\text{core}_k(a)$ la distancia de $a$ a su $k$-ésimo vecino más cercano. Definimos:
               $$d_{\text{mreach-}k}(a, b) = \max\{\text{core}_k(a),\; \text{core}_k(b),\; d(a, b)\}$$
               Los puntos densos permanecen cerca, mientras que los puntos aislados en zonas dispersas son "empujados" lejos automáticamente.
            2. **Árbol de Expansión Mínima (MST):** Se construye el árbol de costo mínimo que conecta los $N$ puntos usando $d_{\text{mreach-}k}$ como peso de cada arista. Cortar las aristas más pesadas equivale a ir subiendo el nivel del mar de forma jerárquica.
            3. **Árbol Condensado (`min_cluster_size`):** Recorremos la jerarquía de arriba hacia abajo. Cuando una división produce una rama menor que `min_cluster_size`, esos puntos **se desprenden como ruido** en lugar de crear micro-clusters espurios.
            4. **Estabilidad de Clusters:** Definimos el nivel de densidad $\lambda = \frac{1}{d_{\text{mreach}}}$. La **estabilidad** de un cluster $C$ mide cuánto persisten sus puntos antes de separarse (visualmente, su **área sombreada** en el árbol condensado):
               $$S(C) = \sum_{p \in C} \left(\lambda_p - \lambda_{\text{birth}}(C)\right)$$
            5. **`cluster_selection_method` (`eom` vs. `leaf`):**
               - **`eom` (*Excess of Mass*):** Maximiza la estabilidad total de abajo hacia arriba: si $S(C) \ge \hat{S}(C_{\text{izq}}) + \hat{S}(C_{\text{der}})$, conserva la isla completa (padre $C$); si no, elige los picos separados (hijos). Produce **grupos más grandes y persistentes**.
               - **`leaf`:** Selecciona siempre los **nodos hoja** (los picos individuales más finos) del fondo del árbol condensado. Produce **grupos más pequeños, finos y homogéneos**.
            6. **`store_centers` (`centroid` vs. `medoid`):**
               - **`centroid`:** Promedio ponderado por probabilidad de los puntos del cluster. Es rápido, pero en formas curvas o no convexas (anillos, medias lunas) **puede caer en el agua (espacio vacío fuera del cluster)**.
               - **`medoid`:** Un **punto real del dataset** dentro del cluster que minimiza la suma de distancias a todos los demás miembros. **Siempre pisa tierra firme (pertenece a los datos reales)**.
            """
        )

# --- Sidebar Navigation ---
st.sidebar.header(tr("1. Choose a Module", "1. Elige un Módulo"))
MODULES = [
    tr("1. Autoencoders: Encoder, Latent Space & Decoder", "1. Autoencoders: Codificador, Espacio Latente y Decodificador"),
    tr("2. Dimensionality Reduction: Autoencoder vs. PCA", "2. Reducción de Dimensionalidad: Autoencoder vs. PCA"),
    tr("3. HDBSCAN Step-by-Step: MST & Cluster Stability", "3. HDBSCAN Paso a Paso: MST y Estabilidad de Clusters"),
    tr("4. HDBSCAN Hyperparameters: eom/leaf & centroid/medoid", "4. Hiperparámetros HDBSCAN: eom/leaf y centroid/medoid"),
    tr("5. HDBSCAN vs. DBSCAN (Variable Densities)", "5. HDBSCAN vs. DBSCAN (Densidades Variables)"),
    tr("6. Real-World Cases & AE + HDBSCAN Pipeline", "6. Casos del Mundo Real y Pipeline AE + HDBSCAN"),
]
mode_idx = st.sidebar.radio(
    tr("Explore Step-by-Step:", "Explora Paso a Paso:"),
    range(len(MODULES)),
    format_func=lambda i: MODULES[i],
)


# ==============================================================================
# MODULE 1: AUTOENCODERS — ENCODER, LATENT SPACE & DECODER
# ==============================================================================
if mode_idx == 0:
    st.subheader(
        tr(
            "Module 1: Inside an Autoencoder — Encoder, Latent Space & Decoder",
            "Módulo 1: Por Dentro de un Autoencoder — Codificador, Espacio Latente y Decodificador",
        )
    )
    st.info(
        tr(
            "**Interactive tip:** Adjust the **Hidden Units**, **Activation**, and **Epochs** in the sidebar to see the bottleneck architecture and reconstruction loss update live. Then move the **(z₁, z₂) sliders** below to watch the **Decoder** synthesize new handwritten digits directly from any point in the 2D Latent Space!",
            "**Sugerencia interactiva:** Ajusta las **Neuronas Ocultas**, la **Activación** y las **Épocas** en la barra lateral para ver cómo cambian la arquitectura y la curva de error. ¡Luego mueve los **controles (z₁, z₂)** abajo para ver cómo el **Decodificador** genera nuevos dígitos manuscritos desde cualquier coordenada del Espacio Latente 2D!",
        )
    )

    st.sidebar.header(tr("Autoencoder Architecture", "Arquitectura del Autoencoder"))
    hidden_dim_m1 = st.sidebar.slider(
        tr("Hidden Layer Units", "Neuronas en Capa Oculta"),
        8,
        48,
        24,
        step=4,
        help=tr(
            "Controls the network's representational capacity: how many intermediate patterns (strokes, loops, edges) the Encoder extracts and the Decoder recombines.",
            "Controla la capacidad de representación de la red: cuántos patrones intermedios (trazos, curvas, bordes) extrae el Codificador y recombina el Decodificador.",
        ),
    )
    act_m1 = st.sidebar.selectbox(
        tr("Hidden Activation Function", "Función de Activación Oculta"),
        ["tanh", "relu", "linear"],
        help=tr(
            "tanh: smooth nonlinear S-curve (-1 to 1). relu: fast piecewise linear (>=0). linear: no nonlinearity (mathematically equivalent to PCA).",
            "tanh: curva S suave no lineal (-1 a 1). relu: lineal a trozos rápida (>=0). linear: sin no linealidad (matemáticamente equivalente a PCA).",
        ),
    )
    epochs_m1 = st.sidebar.slider(tr("Training Epochs", "Épocas de Entrenamiento"), 40, 300, 180, step=20)
    digit_subset_opt = st.sidebar.radio(
        tr("Digit Classes to Train On:", "Clases de Dígitos para Entrenar:"),
        ["5_digits", "all_10"],
        format_func=lambda v: tr("5 Clear Digits (0, 1, 3, 4, 7)", "5 Dígitos Claros (0, 1, 3, 4, 7)")
        if v == "5_digits"
        else tr("All 10 Digits (0–9)", "Todos los 10 Dígitos (0–9)"),
    )

    # Load 8x8 Handwritten Digits (64D -> 2D Latent Space -> 64D)
    digits_raw = load_digits()
    X_dig_all = digits_raw.data / 16.0  # Scale pixel intensities to [0, 1]
    y_dig_all = digits_raw.target
    if digit_subset_opt == "5_digits":
        mask_sub = np.isin(y_dig_all, [0, 1, 3, 4, 7])
        X_dig = X_dig_all[mask_sub][:450]
        y_dig = y_dig_all[mask_sub][:450]
    else:
        X_dig = X_dig_all[:550]
        y_dig = y_dig_all[:550]

    ae_m1 = train_cached_autoencoder(
        X_dig,
        None,
        hidden_dim=hidden_dim_m1,
        latent_dim=2,
        activation=act_m1,
        epochs=epochs_m1,
        lr=0.012,
        batch_size=64,
        seed=42,
    )
    Z_m1 = ae_m1.encode(X_dig)
    X_hat_m1 = np.clip(ae_m1.decode(Z_m1), 0.0, 1.0)
    final_mse = ae_m1.loss_history[-1]
    compression_ratio = X_dig.shape[1] / 2.0

    # Top metrics
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric(tr("Input Dimension (D)", "Dimensión de Entrada (D)"), "64 (8×8 px)")
    mc2.metric(tr("Latent Dimension (d)", "Dimensión Latente (d)"), "2D (z₁, z₂)")
    mc3.metric(tr("Compression Ratio", "Ratio de Compresión"), f"{compression_ratio:.0f}×")
    mc4.metric(tr("Final Reconstruction MSE", "MSE de Reconstrucción Final"), f"{final_mse:.4f}")

    # Architecture diagram + Loss Curve
    col_arch, col_loss = st.columns([3, 2])
    with col_arch:
        fig_arch = plot_autoencoder_architecture(ae_m1.layer_dims, act_m1, tr)
        st.plotly_chart(fig_arch, width="stretch")
    with col_loss:
        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=np.arange(1, len(ae_m1.loss_history) + 1),
                y=ae_m1.loss_history,
                mode="lines",
                name="MSE Loss",
                line=dict(color="#636EFA", width=2.8),
            )
        )
        fig_loss.update_layout(
            title=tr("Training Curve: Reconstruction Loss (MSE)", "Curva de Entrenamiento: Pérdida de Reconstrucción (MSE)"),
            xaxis_title=tr("Epoch", "Época"),
            yaxis_title="MSE",
            height=360,
            template="plotly_white",
        )
        st.plotly_chart(fig_loss, width="stretch")

    # Architecture Guide: Hidden Units & Activation Functions (tanh vs. relu vs. linear)
    with st.expander(
        tr(
            "What Do 'Hidden Layer Units' and 'Activation Functions' (tanh, relu, linear) Do?",
            "¿Qué Hacen las 'Neuronas en Capa Oculta' y las 'Funciones de Activación' (tanh, relu, linear)?",
        ),
        expanded=True,
    ):
        g_col1, g_col2 = st.columns([1.45, 1.0])
        with g_col1:
            if not ES:
                st.markdown(
                    r"""
                    #### 1. Hidden Layer Units (Capacity of the Encoder & Decoder)
                    The **hidden layers** sit between the 64-pixel input/output and the 2D bottleneck. Each hidden neuron learns an intermediate visual feature (a curved stroke, a vertical line, a loop, a corner):
                    - **Few units (e.g., 8):** Low capacity $\to$ *underfitting* (blurry digits, higher MSE).
                    - **Balanced units (e.g., 16–32):** Enough expressive power to compress shapes cleanly into 2D.
                    - **Many units (e.g., 48):** Higher capacity and lower training MSE, but slower and more prone to memorizing noise (*overfitting*).

                    #### 2. Hidden Activation Functions (`tanh` vs. `relu` vs. `linear`)
                    | Activation | Formula | Behavior in an Autoencoder |
                    | :--- | :--- | :--- |
                    | **`tanh`** | $\frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$ | **Smooth, zero-centered S-curve.** Creates smooth, continuous latent spaces where moving between two digits morphs one gently into the other. |
                    | **`relu`** | $\max(0, x) \in [0, \infty)$ | **Piecewise linear & fast.** Turns negative activations to 0 (sparse features); great for sharp strokes, though some neurons can become inactive ("dead ReLU"). |
                    | **`linear`** | $f(x) = x$ | **No nonlinearity!** Stacking linear layers collapses into a single matrix multiplication — mathematically equivalent to **PCA** (higher MSE on curved data). |
                    """
                )
            else:
                st.markdown(
                    r"""
                    #### 1. Neuronas en Capa Oculta (Capacidad del Codificador y Decodificador)
                    Las **capas ocultas** se ubican entre la entrada/salida de 64 píxeles y el cuello de botella 2D. Cada neurona oculta aprende un rasgo visual intermedio (un trazo curvo, una línea vertical, un bucle, una esquina):
                    - **Pocas neuronas (ej. 8):** Baja capacidad $\to$ *subajuste (underfitting)*: dígitos borrosos y mayor MSE.
                    - **Cantidad equilibrada (ej. 16–32):** Suficiente expresividad para comprimir las formas limpiamente en 2D.
                    - **Muchas neuronas (ej. 48):** Mayor capacidad y menor MSE en entrenamiento, pero más lento y con mayor riesgo de memorizar ruido (*sobreajuste / overfitting*).

                    #### 2. Funciones de Activación Oculta (`tanh` vs. `relu` vs. `linear`)
                    | Activación | Fórmula | Comportamiento en un Autoencoder |
                    | :--- | :--- | :--- |
                    | **`tanh`** | $\frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)$ | **Curva en S suave y centrada en cero.** Produce espacios latentes continuos y suaves donde moverse entre dos dígitos transforma uno gradualmente en el otro. |
                    | **`relu`** | $\max(0, x) \in [0, \infty)$ | **Lineal a trozos y rápida.** Apaga valores negativos en 0 (representaciones dispersas); captura bien trazos definidos, aunque algunas neuronas pueden "apagarse" (*dead ReLU*). |
                    | **`linear`** | $f(x) = x$ | **Sin no linealidad.** Apilar capas lineales colapsa en una sola multiplicación matricial: es matemáticamente equivalente a **PCA** (mayor MSE en patrones curvos). |
                    """
                )
        with g_col2:
            xs_act = np.linspace(-3.0, 3.0, 200)
            fig_act = go.Figure()
            for name_a, ys_a, col_a in [
                ("tanh", np.tanh(xs_act), "#00CC96"),
                ("relu", np.maximum(0.0, xs_act), "#EF553B"),
                ("linear", xs_act, "#636EFA"),
            ]:
                is_cur = name_a == act_m1
                fig_act.add_trace(
                    go.Scatter(
                        x=xs_act,
                        y=ys_a,
                        mode="lines",
                        name=f"{name_a} {'(active)' if is_cur and not ES else '(activa)' if is_cur else ''}",
                        line=dict(color=col_a, width=4.0 if is_cur else 1.8, dash="solid" if is_cur else "dot"),
                    )
                )
            fig_act.add_hline(y=0, line_color="#9CA3AF", line_width=1)
            fig_act.add_vline(x=0, line_color="#9CA3AF", line_width=1)
            fig_act.update_layout(
                title=tr(f"Activation Curves (Selected: {act_m1})", f"Curvas de Activación (Activa: {act_m1})"),
                xaxis_title="x",
                yaxis_title="f(x)",
                yaxis=dict(range=[-1.5, 3.2]),
                height=340,
                template="plotly_white",
                legend=dict(orientation="h", y=-0.22),
            )
            st.plotly_chart(fig_act, width="stretch")

    st.markdown("---")
    st.markdown(
        tr(
            "### Interactive Latent Space & Live Decoder Playground",
            "### Explorador Interactivo del Espacio Latente y Decodificador en Vivo",
        )
    )

    tab_dec, tab_rec = st.tabs(
        [
            tr("1. Navigate Latent Space (z₁, z₂) → Decode New Digit", "1. Navegar Espacio Latente (z₁, z₂) → Decodificar Nuevo Dígito"),
            tr("2. Inspect Original vs. Encoded vs. Reconstructed Samples", "2. Inspeccionar Muestras: Original vs. Código Latente vs. Reconstrucción"),
        ]
    )

    with tab_dec:
        st.caption(
            tr(
                "How to read this: Each colored dot in the 2D map is a **real 8×8 handwritten digit** compressed by the Encoder to 2 coordinates (z₁, z₂). Move the **gold star ★** anywhere on the map: on the right you will see both the **nearest real original digit** from the dataset AND the **image synthesized by the Decoder** at your exact star coordinates!",
                "Cómo leer esto: Cada punto de color en el mapa 2D es un **dígito manuscrito real de 8×8 píxeles** que el Codificador comprimió a 2 coordenadas (z₁, z₂). Mueve la **estrella dorada ★** por el mapa: a la derecha verás tanto el **dígito original real más cercano** del dataset como la **imagen que genera el Decodificador** en la coordenada exacta de tu estrella.",
            )
        )
        ld1, ld2, ld3 = st.columns([1.05, 1.85, 1.45])
        z1_min, z1_max = float(np.percentile(Z_m1[:, 0], 1)), float(np.percentile(Z_m1[:, 0], 99))
        z2_min, z2_max = float(np.percentile(Z_m1[:, 1], 1)), float(np.percentile(Z_m1[:, 1], 99))
        with ld1:
            st.markdown(
                tr(
                    "Move **$(z_1, z_2)$** inside the 2D Latent Space to test what the **Decoder $g_\\theta(z)$** draws:",
                    "Mueve **$(z_1, z_2)$** dentro del Espacio Latente 2D para ver qué dibuja el **Decodificador $g_\\theta(z)$**:",
                )
            )
            z1_val = st.slider("z₁", z1_min, z1_max, float(np.median(Z_m1[:, 0])), step=0.1)
            z2_val = st.slider("z₂", z2_min, z2_max, float(np.median(Z_m1[:, 1])), step=0.1)

            # Find nearest real original sample in latent space
            dists_z = np.linalg.norm(Z_m1 - np.array([z1_val, z2_val]), axis=1)
            nearest_i = int(np.argmin(dists_z))
            nearest_dist = float(dists_z[nearest_i])
            nearest_digit = int(y_dig[nearest_i])

            st.metric(
                tr("Nearest Real Sample in Map", "Muestra Real Más Cercana"),
                tr(f"Digit {nearest_digit} (#{nearest_i})", f"Dígito {nearest_digit} (#{nearest_i})"),
                tr(f"Latent dist: {nearest_dist:.2f}", f"Dist. latente: {nearest_dist:.2f}"),
            )
            st.caption(
                tr(
                    "Place the star inside a color cluster to decode that digit cleanly, or halfway between two clusters to watch one digit morph into another!",
                    "Coloca la estrella dentro de un grupo de color para decodificar ese dígito limpiamente, ¡o a mitad de camino entre dos grupos para ver cómo un dígito se transforma en otro!",
                )
            )

        with ld2:
            fig_lat = px.scatter(
                x=Z_m1[:, 0],
                y=Z_m1[:, 1],
                color=y_dig.astype(str),
                title=tr("2D Latent Space z = Encoder(x)", "Espacio Latente 2D z = Codificador(x)"),
                labels={"x": "z₁", "y": "z₂", "color": tr("Digit", "Dígito")},
            )
            # Dashed line connecting probe to nearest real sample
            fig_lat.add_trace(
                go.Scatter(
                    x=[z1_val, Z_m1[nearest_i, 0]],
                    y=[z2_val, Z_m1[nearest_i, 1]],
                    mode="lines",
                    line=dict(color="#111827", width=2, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Highlight nearest real sample
            fig_lat.add_trace(
                go.Scatter(
                    x=[Z_m1[nearest_i, 0]],
                    y=[Z_m1[nearest_i, 1]],
                    mode="markers",
                    name=tr(f"Nearest real (Digit {nearest_digit})", f"Real más cercano (Dígito {nearest_digit})"),
                    marker=dict(size=14, symbol="circle-open", color="#111827", line=dict(width=3, color="#111827")),
                )
            )
            fig_lat.add_trace(
                go.Scatter(
                    x=[z1_val],
                    y=[z2_val],
                    mode="markers",
                    name=tr("Your probe (z₁, z₂)", "Tu punto (z₁, z₂)"),
                    marker=dict(size=18, symbol="star", color="gold", line=dict(width=2, color="black")),
                )
            )
            fig_lat.update_layout(height=440, template="plotly_white", legend=dict(orientation="h", y=-0.22))
            st.plotly_chart(fig_lat, width="stretch")

        with ld3:
            decoded_img = np.clip(ae_m1.decode(np.array([[z1_val, z2_val]])), 0.0, 1.0).reshape(8, 8)
            nearest_orig_img = X_dig[nearest_i].reshape(8, 8)

            fig_near = px.imshow(
                nearest_orig_img,
                color_continuous_scale="gray_r",
                zmin=0.0,
                zmax=1.0,
                title=tr(f"Nearest Original Input x (Real Digit {nearest_digit})", f"Entrada Original Más Cercana x (Dígito Real {nearest_digit})"),
            )
            fig_near.update_coloraxes(showscale=False)
            fig_near.update_xaxes(showticklabels=False)
            fig_near.update_yaxes(showticklabels=False)
            fig_near.update_layout(height=210, margin=dict(l=10, r=10, t=38, b=5))
            st.plotly_chart(fig_near, width="stretch")

            fig_gen = px.imshow(
                decoded_img,
                color_continuous_scale="gray_r",
                zmin=0.0,
                zmax=1.0,
                title=tr(f"Decoder Output g({z1_val:.1f}, {z2_val:.1f})", f"Salida del Decodificador g({z1_val:.1f}, {z2_val:.1f})"),
            )
            fig_gen.update_coloraxes(showscale=False)
            fig_gen.update_xaxes(showticklabels=False)
            fig_gen.update_yaxes(showticklabels=False)
            fig_gen.update_layout(height=210, margin=dict(l=10, r=10, t=38, b=5))
            st.plotly_chart(fig_gen, width="stretch")

    with tab_rec:
        sample_idx = st.slider(tr("Select sample index to inspect:", "Selecciona el índice de muestra a inspeccionar:"), 0, len(X_dig) - 1, 0)
        rc1, rc2, rc3 = st.columns([1.2, 1.0, 1.2])
        orig_img = X_dig[sample_idx].reshape(8, 8)
        rec_img = X_hat_m1[sample_idx].reshape(8, 8)
        z_pt = Z_m1[sample_idx]
        pt_mse = float(np.mean((orig_img - rec_img) ** 2))

        with rc1:
            fig_o = px.imshow(orig_img, color_continuous_scale="gray_r", zmin=0, zmax=1, title=tr(f"Original Input x (Digit {y_dig[sample_idx]})", f"Entrada Original x (Dígito {y_dig[sample_idx]})"))
            fig_o.update_coloraxes(showscale=False)
            fig_o.update_xaxes(showticklabels=False)
            fig_o.update_yaxes(showticklabels=False)
            fig_o.update_layout(height=320)
            st.plotly_chart(fig_o, width="stretch")
        with rc2:
            st.markdown(f"#### {tr('Compressed Code z', 'Código Comprimido z')}")
            st.latex(rf"z = \begin{{bmatrix}} {z_pt[0]:.3f} \\ {z_pt[1]:.3f} \end{{bmatrix}} \in \mathbb{{R}}^2")
            st.metric(tr("Sample MSE", "MSE de esta Muestra"), f"{pt_mse:.4f}")
            st.caption(
                tr(
                    "64 pixel numbers were compressed down to just these 2 numbers, and then expanded back to 64 pixels.",
                    "64 valores de píxeles se comprimieron en solo estos 2 números y luego se expandieron de vuelta a 64 píxeles.",
                )
            )
        with rc3:
            fig_r = px.imshow(rec_img, color_continuous_scale="gray_r", zmin=0, zmax=1, title=tr("Reconstructed x̂ = Decoder(Encoder(x))", "Reconstrucción x̂ = Decodificador(Codificador(x))"))
            fig_r.update_coloraxes(showscale=False)
            fig_r.update_xaxes(showticklabels=False)
            fig_r.update_yaxes(showticklabels=False)
            fig_r.update_layout(height=320)
            st.plotly_chart(fig_r, width="stretch")


# ==============================================================================
# MODULE 2: DIMENSIONALITY REDUCTION — AUTOENCODER VS. PCA
# ==============================================================================
elif mode_idx == 1:
    st.subheader(
        tr(
            "Module 2: Dimensionality Reduction — Nonlinear Autoencoder vs. Linear PCA",
            "Módulo 2: Reducción de Dimensionalidad — Autoencoder No Lineal vs. PCA Lineal",
        )
    )
    st.info(
        tr(
            "**Key insight:** PCA projects data onto a **flat linear hyperplane**. A **nonlinear Autoencoder** (`tanh` / `relu`) bends its latent coordinate sheet to fit curved manifolds! Try switching the Autoencoder activation to **`linear`** in the sidebar — you will see its Reconstruction MSE match PCA almost identically!",
            "**Idea clave:** PCA proyecta los datos sobre un **hiperplano lineal plano**. Un **Autoencoder no lineal** (`tanh` / `relu`) dobla su superficie latente para adaptarse a variedades curvas. ¡Prueba cambiar la activación del Autoencoder a **`linear`** en la barra lateral y verás cómo su MSE iguala casi exactamente al de PCA!",
        )
    )

    st.sidebar.header(tr("Dimensionality Reduction Controls", "Controles de Reducción"))
    MANIFOLDS = [
        ("swiss_roll", tr("3D Swiss Roll (3D → 2D)", "Rollo Suizo 3D (3D → 2D)")),
        ("s_curve", tr("3D S-Curve Manifold (3D → 2D)", "Curva S en 3D (3D → 2D)")),
        ("helix", tr("3D Intertwined Helices + Core (3D → 2D)", "Hélices 3D Entrelazadas + Núcleo (3D → 2D)")),
        ("digits", tr("Handwritten Digits (64D → 2D)", "Dígitos Manuscritos (64D → 2D)")),
    ]
    mf_idx = st.sidebar.selectbox(
        tr("Dataset / Manifold:", "Dataset / Variedad:"),
        range(len(MANIFOLDS)),
        format_func=lambda i: MANIFOLDS[i][1],
    )
    mf_key = MANIFOLDS[mf_idx][0]
    ae_act_m2 = st.sidebar.selectbox(tr("Autoencoder Activation:", "Activación del Autoencoder:"), ["tanh", "relu", "linear"])
    ae_hid_m2 = st.sidebar.slider(tr("Autoencoder Hidden Units", "Neuronas Ocultas del Autoencoder"), 8, 48, 24, step=4)
    ae_ep_m2 = st.sidebar.slider(tr("Training Epochs", "Épocas de Entrenamiento"), 80, 400, 260, step=20)

    if mf_key == "digits":
        X_m2, color_m2, _ = load_real_highd_dataset("digits", max_samples=450, seed=42)
        is_3d = False
    else:
        X_m2, color_m2 = make_3d_manifold(mf_key, n_samples=360, noise=0.06, seed=42)
        is_3d = True

    # Fit PCA (2D)
    pca = PCA(n_components=2, random_state=42)
    Z_pca = pca.fit_transform(X_m2)
    X_hat_pca = pca.inverse_transform(Z_pca)
    mse_pca = float(np.mean((X_m2 - X_hat_pca) ** 2))
    var_pca = float(np.sum(pca.explained_variance_ratio_) * 100)

    # Fit Autoencoder (2D)
    ae_m2 = train_cached_autoencoder(
        X_m2,
        None,
        hidden_dim=ae_hid_m2,
        latent_dim=2,
        activation=ae_act_m2,
        epochs=ae_ep_m2,
        lr=0.015,
        batch_size=64,
        seed=42,
    )
    Z_ae = ae_m2.encode(X_m2)
    X_hat_ae = ae_m2.decode(Z_ae)
    mse_ae = float(np.mean((X_m2 - X_hat_ae) ** 2))
    mse_gain = (mse_pca - mse_ae) / max(mse_pca, 1e-9) * 100.0

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr("PCA Reconstruction MSE", "MSE Reconstrucción PCA"), f"{mse_pca:.4f}", f"{var_pca:.1f}% var")
    c2.metric(
        tr(f"Autoencoder ({ae_act_m2}) MSE", f"MSE Autoencoder ({ae_act_m2})"),
        f"{mse_ae:.4f}",
        f"{mse_gain:+.1f}% vs PCA",
    )
    if mf_key in ("digits", "helix"):
        y_cls = color_m2.astype(int)
        knn_pca = KNeighborsClassifier(n_neighbors=5).fit(Z_pca, y_cls).score(Z_pca, y_cls)
        knn_ae = KNeighborsClassifier(n_neighbors=5).fit(Z_ae, y_cls).score(Z_ae, y_cls)
        c3.metric(tr("PCA 2D Local Purity (5-NN)", "Pureza Local PCA 2D (5-NN)"), f"{knn_pca*100:.1f}%")
        c4.metric(tr("AE 2D Local Purity (5-NN)", "Pureza Local AE 2D (5-NN)"), f"{knn_ae*100:.1f}%")
    else:
        c3.metric(tr("Original Dimension", "Dimensión Original"), f"{X_m2.shape[1]}D")
        c4.metric(tr("Compressed Dimension", "Dimensión Comprimida"), "2D")

    if is_3d:
        col_3d, col_rec3d = st.columns(2)
        with col_3d:
            fig_orig3d = px.scatter_3d(
                x=X_m2[:, 0],
                y=X_m2[:, 1],
                z=X_m2[:, 2],
                color=color_m2,
                color_continuous_scale="Turbo",
                title=tr("Original 3D Nonlinear Manifold", "Variedad No Lineal Original en 3D"),
            )
            fig_orig3d.update_traces(marker=dict(size=4))
            fig_orig3d.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_orig3d, width="stretch")

        with col_rec3d:
            rec_choice = st.radio(
                tr("Compare 3D Reconstruction:", "Comparar Reconstrucción 3D:"),
                ["ae", "pca"],
                horizontal=True,
                format_func=lambda v: tr(f"Autoencoder ({ae_act_m2}) Reconstruction (MSE={mse_ae:.4f})", f"Reconstrucción Autoencoder ({ae_act_m2}) (MSE={mse_ae:.4f})")
                if v == "ae"
                else tr(f"PCA Flat Plane Reconstruction (MSE={mse_pca:.4f})", f"Reconstrucción Plano PCA (MSE={mse_pca:.4f})"),
            )
            X_show = X_hat_ae if rec_choice == "ae" else X_hat_pca
            fig_rec3d = px.scatter_3d(
                x=X_show[:, 0],
                y=X_show[:, 1],
                z=X_show[:, 2],
                color=color_m2,
                color_continuous_scale="Turbo",
                title=tr("Reconstructed 3D Surface from 2D Code", "Superficie 3D Reconstruida desde Código 2D"),
            )
            fig_rec3d.update_traces(marker=dict(size=4))
            fig_rec3d.update_layout(height=370, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_rec3d, width="stretch")

    # Side-by-side 2D Embeddings: PCA vs Autoencoder
    p1, p2 = st.columns(2)
    is_discrete = mf_key in ("digits", "helix")
    color_arg = color_m2.astype(int).astype(str) if is_discrete else color_m2
    with p1:
        fig_pca2d = px.scatter(
            x=Z_pca[:, 0],
            y=Z_pca[:, 1],
            color=color_arg,
            color_continuous_scale="Turbo" if not is_discrete else None,
            title=tr(f"PCA 2D Projection (Linear — MSE={mse_pca:.4f})", f"Proyección PCA 2D (Lineal — MSE={mse_pca:.4f})"),
            labels={"x": "PC 1", "y": "PC 2", "color": tr("Class / Pos", "Clase / Pos")},
        )
        fig_pca2d.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig_pca2d, width="stretch")
    with p2:
        fig_ae2d = px.scatter(
            x=Z_ae[:, 0],
            y=Z_ae[:, 1],
            color=color_arg,
            color_continuous_scale="Turbo" if not is_discrete else None,
            title=tr(f"Autoencoder 2D Latent Space ({ae_act_m2} — MSE={mse_ae:.4f})", f"Espacio Latente 2D Autoencoder ({ae_act_m2} — MSE={mse_ae:.4f})"),
            labels={"x": "z₁", "y": "z₂", "color": tr("Class / Pos", "Clase / Pos")},
        )
        fig_ae2d.update_layout(height=420, template="plotly_white")
        st.plotly_chart(fig_ae2d, width="stretch")

    # Summary Table PCA vs Autoencoder
    if not ES:
        st.markdown(
            """
            | Criterion | **PCA** | **Nonlinear Autoencoder** |
            | :--- | :--- | :--- |
            | **Transformation Type** | Strictly linear orthogonal projection ($Z = XW$) | Nonlinear neural mapping ($z = f_\\phi(x)$, $\\hat{x} = g_\\theta(z)$) |
            | **Curved Manifolds** | Flattens/crushes curved geometry onto a plane | Bends & unrolls nonlinear manifolds |
            | **Deterministic / Closed-Form?** | Yes (exact SVD eigendecomposition) | No (gradient descent; depends on init & hyperparameters) |
            | **When Activation = `linear`** | Exact global optimum for linear MSE | Spans the **exact same subspace** as PCA! |
            """
        )
    else:
        st.markdown(
            """
            | Criterio | **PCA** | **Autoencoder No Lineal** |
            | :--- | :--- | :--- |
            | **Tipo de Transformación** | Proyección lineal ortogonal estricta ($Z = XW$) | Mapeo neuronal no lineal ($z = f_\\phi(x)$, $\\hat{x} = g_\\theta(z)$) |
            | **Geometrías Curvas (*Manifolds*)** | Aplasta la curvatura sobre un plano rígido | Se dobla y desenrolla variedades no lineales |
            | **¿Solución Exacta / Determinista?** | Sí (descomposición SVD exacta) | No (descenso de gradiente; depende de inicialización y épocas) |
            | **Si la activación es `linear`** | Óptimo global exacto para MSE lineal | ¡Aprende **exactamente el mismo subespacio** que PCA! |
            """
        )


# ==============================================================================
# MODULE 3: HDBSCAN STEP-BY-STEP — CORE DIST, MST, CONDENSED TREE & STABILITY
# ==============================================================================
elif mode_idx == 2:
    st.subheader(
        tr(
            "Module 3: HDBSCAN Step-by-Step — Mutual Reachability, MST & Cluster Stability",
            "Módulo 3: HDBSCAN Paso a Paso — Alcanzabilidad Mutua, MST y Estabilidad de Clusters",
        )
    )
    st.info(
        tr(
            "**Follow the 4 internal stages of HDBSCAN:** (1) Transform distances with **Core Distance & Mutual Reachability**, (2) Build & cut the **Minimum Spanning Tree (MST)**, (3) Condense the tree & compute **Cluster Stability**, and (4) Extract final clusters with **soft membership probabilities**.",
            "**Sigue las 4 etapas internas de HDBSCAN:** (1) Transforma distancias con **Distancia Núcleo y Alcanzabilidad Mutua**, (2) Construye y corta el **Árbol de Expansión Mínima (MST)**, (3) Condensa el árbol y calcula la **Estabilidad de Clusters**, y (4) Extrae los grupos finales con **probabilidades de pertenencia**.",
        )
    )

    st.sidebar.header(tr("HDBSCAN Controls", "Controles de HDBSCAN"))
    PAT_HDB = [
        ("hierarchical", tr("Hierarchical Multi-Scale Clusters", "Clusters Jerárquicos Multiescala")),
        ("blobs", tr("Varying Densities (Dense + Sparse)", "Densidades Variables (Densos + Disperso)")),
        ("non_convex_mixed", tr("Dense Moons + Sparse Cloud", "Medias Lunas Densas + Nube Dispersa")),
    ]
    pat_h_idx = st.sidebar.selectbox(
        tr("Dataset:", "Conjunto de Datos:"),
        range(len(PAT_HDB)),
        format_func=lambda i: PAT_HDB[i][1],
    )
    mcs_m3 = st.sidebar.slider("min_cluster_size", 5, 45, 15, step=1)
    ms_m3 = st.sidebar.slider("min_samples (k)", 2, 20, 5, step=1)
    n_noise_m3 = st.sidebar.slider(tr("Background Noise Points", "Puntos de Ruido de Fondo"), 0, 40, 20, step=5)

    X_m3, y_m3 = make_varying_density_dataset(PAT_HDB[pat_h_idx][0], n_noise=n_noise_m3, seed=42)
    hres = cached_hdbscan_steps(X_m3, min_cluster_size=mcs_m3, min_samples=ms_m3)

    step_tabs = st.tabs(
        [
            tr("Step 1: Core & Mutual Reachability", "Paso 1: Distancia Núcleo y Alcanzabilidad Mutua"),
            tr("Step 2: Minimum Spanning Tree (MST)", "Paso 2: Árbol de Expansión Mínima (MST)"),
            tr("Step 3: Condensed Tree & Stability", "Paso 3: Árbol Condensado y Estabilidad"),
            tr("Step 4: Final Clusters & Probabilities", "Paso 4: Clusters Finales y Probabilidades"),
        ]
    )

    # ---- Step 1: Core Distance & Mutual Reachability ----
    with step_tabs[0]:
        s1_c1, s1_c2 = st.columns([1.1, 2.0])
        core_d = hres["core_dists"]
        with s1_c1:
            st.markdown(
                tr(
                    r"""
                    #### Why transform space first?
                    In raw Euclidean space, a noisy point sitting between two clusters can accidentally bridge them.
                    HDBSCAN computes each point's **Core Distance** $\text{core}_k(a)$ (distance to its $k$-th nearest neighbor, where $k=$ `min_samples`), then defines **Mutual Reachability Distance**:
                    $$d_{\text{mreach-}k}(a, b) = \max\{\text{core}_k(a),\; \text{core}_k(b),\; d(a, b)\}$$
                    - **Dense points** have tiny $\text{core}_k$ $\to$ distances stay small.
                    - **Sparse / noisy points** have large $\text{core}_k$ $\to$ they get **pushed away** from everyone!
                    """,
                    r"""
                    #### ¿Por qué transformar el espacio primero?
                    En el espacio euclidiano crudo, un punto de ruido entre dos grupos puede crear un puente accidental.
                    HDBSCAN calcula la **Distancia Núcleo** $\text{core}_k(a)$ (distancia al $k$-ésimo vecino más cercano, con $k=$ `min_samples`) y define la **Distancia de Alcanzabilidad Mutua**:
                    $$d_{\text{mreach-}k}(a, b) = \max\{\text{core}_k(a),\; \text{core}_k(b),\; d(a, b)\}$$
                    - **Puntos densos** tienen $\text{core}_k$ pequeño $\to$ siguen cerca entre sí.
                    - **Puntos dispersos / ruido** tienen $\text{core}_k$ grande $\to$ son **alejados** automáticamente de todos.
                    """,
                )
            )
            pt_idx = st.slider(tr("Inspect Point # (shows core_k circle):", "Inspeccionar Punto # (muestra círculo core_k):"), 0, len(X_m3) - 1, 0)
            st.metric(tr(f"Core Distance core_{ms_m3}(p)", f"Distancia Núcleo core_{ms_m3}(p)"), f"{core_d[pt_idx]:.3f}")

        with s1_c2:
            fig_core = go.Figure()
            fig_core.add_trace(
                go.Scatter(
                    x=X_m3[:, 0],
                    y=X_m3[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=core_d,
                        colorscale="YlOrRd",
                        colorbar=dict(title=f"core_{ms_m3}"),
                        line=dict(width=0.5, color="#1F2937"),
                    ),
                    text=[f"#{i} | core={core_d[i]:.3f}" for i in range(len(X_m3))],
                    hoverinfo="text",
                    name=tr("Points colored by core_k", "Puntos coloreados por core_k"),
                )
            )
            # Draw core_k circle around inspected point
            theta = np.linspace(0, 2 * np.pi, 80)
            r_c = core_d[pt_idx]
            fig_core.add_trace(
                go.Scatter(
                    x=X_m3[pt_idx, 0] + r_c * np.cos(theta),
                    y=X_m3[pt_idx, 1] + r_c * np.sin(theta),
                    mode="lines",
                    line=dict(color="#2563EB", width=2.2, dash="dash"),
                    name=tr(f"core_{ms_m3} radius = {r_c:.2f}", f"Radio core_{ms_m3} = {r_c:.2f}"),
                )
            )
            fig_core.add_trace(
                go.Scatter(
                    x=[X_m3[pt_idx, 0]],
                    y=[X_m3[pt_idx, 1]],
                    mode="markers",
                    marker=dict(size=15, symbol="star", color="gold", line=dict(width=1.5, color="black")),
                    name=tr(f"Point #{pt_idx}", f"Punto #{pt_idx}"),
                )
            )
            fig_core.update_layout(
                title=tr("Points Colored by Core Distance (Red = Sparse/Noise, Yellow = Dense Core)", "Puntos Coloreados por Distancia Núcleo (Rojo = Disperso/Ruido, Amarillo = Núcleo Denso)"),
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=470,
                template="plotly_white",
                legend=dict(orientation="h", y=-0.18),
            )
            st.plotly_chart(fig_core, width="stretch")

    # ---- Step 2: Minimum Spanning Tree (MST) ----
    with step_tabs[1]:
        mst_edges = hres["mst_edges"]
        weights = np.array([e[2] for e in mst_edges])
        w_min, w_max = float(weights.min()), float(weights.max())

        m2_c1, m2_c2 = st.columns([1.05, 2.1])
        with m2_c1:
            st.markdown(
                tr(
                    """
                    #### Minimum Spanning Tree (MST)
                    Think of the dataset as islands connected by bridges weighted by $d_{\\text{mreach}}$.
                    The **MST** keeps only the **$N - 1$ cheapest bridges** needed to connect all points without cycles (using Prim's algorithm).

                    **Slide the threshold below** to cut the heaviest MST edges (equivalent to raising the density level $\\lambda = 1/w$) and watch clusters emerge!
                    """,
                    """
                    #### Árbol de Expansión Mínima (MST)
                    Imagina los datos como islas conectadas por puentes cuyo peso es $d_{\\text{mreach}}$.
                    El **MST** conserva únicamente los **$N - 1$ puentes más baratos** que conectan todos los puntos sin formar ciclos (algoritmo de Prim).

                    **Mueve el deslizador abajo** para cortar las aristas más pesadas del MST (equivalente a subir el nivel de densidad $\\lambda = 1/w$) ¡y observa cómo emergen los clusters!
                    """,
                )
            )
            default_thr = float(np.percentile(weights, 86))
            mst_thr = st.slider(
                tr("Cut MST edges with weight > threshold:", "Cortar aristas MST con peso > umbral:"),
                round(w_min, 3),
                round(w_max + 0.05, 3),
                round(default_thr, 3),
                step=0.02,
            )
            cut_labels = mst_connected_components(len(X_m3), mst_edges, mst_thr, min_size=mcs_m3)
            n_comp = len(set(cut_labels) - {-1})
            n_cut_edges = int(np.sum(weights > mst_thr))
            st.metric(tr("Cut MST Edges (Red Dashed)", "Aristas MST Cortadas (Rojo Punteado)"), f"{n_cut_edges} / {len(weights)}")
            st.metric(tr(f"Clusters (size ≥ {mcs_m3})", f"Clusters (tamaño ≥ {mcs_m3})"), f"{n_comp}")
            st.metric(tr("Equivalent Density λ = 1 / w", "Densidad Equivalente λ = 1 / w"), f"{1.0 / max(mst_thr, 1e-6):.2f}")

            # Histogram of MST edge weights
            fig_wh = px.histogram(
                x=weights,
                nbins=30,
                title=tr("MST Edge Weight Distribution", "Distribución de Pesos de Aristas del MST"),
                labels={"x": "d_mreach"},
            )
            fig_wh.add_vline(x=mst_thr, line_dash="dash", line_color="red")
            fig_wh.update_layout(height=220, margin=dict(l=10, r=10, t=35, b=10), template="plotly_white")
            st.plotly_chart(fig_wh, width="stretch")

        with m2_c2:
            fig_mst = plot_mst_graph(X_m3, mst_edges, mst_thr, cut_labels, tr)
            st.plotly_chart(fig_mst, width="stretch")

    # ---- Step 3: Condensed Tree & Cluster Stability ----
    with step_tabs[2]:
        condensed = hres["condensed"]
        eom_sel = hres["eom_selected"]
        leaf_sel = hres["leaf_selected"]

        st_c1, st_c2 = st.columns([2.0, 1.15])
        with st_c1:
            tree_mode = st.radio(
                tr("Highlight selection on Condensed Tree:", "Resaltar selección en el Árbol Condensado:"),
                ["eom", "leaf"],
                horizontal=True,
                format_func=lambda m: "EOM (Excess of Mass)" if m == "eom" else "Leaf",
                key="m3_tree_mode",
            )
            fig_ct = plot_condensed_tree(condensed, eom_sel, leaf_sel, tree_mode, tr)
            st.plotly_chart(fig_ct, width="stretch")

        with st_c2:
            st.markdown(
                tr(
                    r"""
                    #### Cluster Stability Formula
                    For every cluster $C$ in the condensed tree:
                    $$S(C) = \sum_{p \in C} \left(\lambda_p - \lambda_{\text{birth}}(C)\right)$$
                    - **Shaded area** of each funnel = **Stability $S(C)$**.
                    - **Bottom-up EOM Rule:**
                      $$\hat{S}(C) = \max\Big(S(C),\; \hat{S}(C_L) + \hat{S}(C_R)\Big)$$
                      If a parent cluster has **more area (stability)** than the sum of its children, EOM picks the **parent**; otherwise it picks the **children**!
                    """,
                    r"""
                    #### Fórmula de Estabilidad de Clusters
                    Para cada cluster $C$ en el árbol condensado:
                    $$S(C) = \sum_{p \in C} \left(\lambda_p - \lambda_{\text{birth}}(C)\right)$$
                    - **Área sombreada** de cada embudo = **Estabilidad $S(C)$**.
                    - **Regla EOM de abajo hacia arriba:**
                      $$\hat{S}(C) = \max\Big(S(C),\; \hat{S}(C_L) + \hat{S}(C_R)\Big)$$
                      Si un cluster padre tiene **más área (estabilidad)** que la suma de sus hijos, EOM elige al **padre**; de lo contrario elige a los **hijos**.
                    """,
                )
            )
            # Table of condensed tree nodes (excluding root 0)
            node_rows = []
            for cid in sorted(condensed.keys()):
                if cid == 0:
                    continue
                inf = condensed[cid]
                node_rows.append(
                    {
                        "Node": f"C{cid}",
                        tr("Points", "Puntos"): inf["initial_size"],
                        "S(C)": round(inf["stability"], 2),
                        tr("Leaf?", "¿Hoja?"): "✓" if inf["is_leaf"] else "",
                        "EOM": "✓" if cid in eom_sel else "",
                        "Leaf": "✓" if cid in leaf_sel else "",
                    }
                )
            if node_rows:
                df_nodes = pd.DataFrame(node_rows)
                st.dataframe(df_nodes, hide_index=True, use_container_width=True)
                fig_sb = px.bar(
                    df_nodes,
                    x="Node",
                    y="S(C)",
                    color="EOM",
                    title=tr("Stability S(C) by Tree Node", "Estabilidad S(C) por Nodo del Árbol"),
                )
                fig_sb.update_layout(height=230, margin=dict(l=10, r=10, t=35, b=10), template="plotly_white", showlegend=False)
                st.plotly_chart(fig_sb, width="stretch")

    # ---- Step 4: Final Clusters & Soft Membership Probabilities ----
    with step_tabs[3]:
        hdb_final = hres["hdb_eom"]
        lbls = hdb_final.labels_
        probs = hdb_final.probabilities_

        f1, f2 = st.columns(2)
        with f1:
            fig_fin = px.scatter(
                x=X_m3[:, 0],
                y=X_m3[:, 1],
                color=format_cluster_labels(lbls),
                color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#9CA3AF"},
                title=tr("Final HDBSCAN Clusters (EOM)", "Clusters Finales HDBSCAN (EOM)"),
                labels={"x": "$x_1$", "y": "$x_2$", "color": "HDBSCAN"},
            )
            fig_fin.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_fin.update_layout(height=440, template="plotly_white", legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig_fin, width="stretch")

        with f2:
            fig_prob = go.Figure()
            fig_prob.add_trace(
                go.Scatter(
                    x=X_m3[:, 0],
                    y=X_m3[:, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=probs,
                        colorscale="Viridis",
                        cmin=0.0,
                        cmax=1.0,
                        colorbar=dict(title="P(cluster)"),
                        line=dict(width=0.5, color="#1F2937"),
                    ),
                    text=[f"Cluster {lbls[i] + 1 if lbls[i] != -1 else 'Noise'} | prob={probs[i]:.2f}" for i in range(len(X_m3))],
                    hoverinfo="text",
                )
            )
            fig_prob.update_layout(
                title=tr("Soft Membership Probabilities (probabilities_)", "Probabilidades Suaves de Pertenencia (probabilities_)"),
                xaxis_title="$x_1$",
                yaxis_title="$x_2$",
                yaxis=dict(scaleanchor="x", scaleratio=1),
                height=440,
                template="plotly_white",
            )
            st.plotly_chart(fig_prob, width="stretch")


# ==============================================================================
# MODULE 4: HDBSCAN HYPERPARAMETERS — cluster_selection_method & store_centers
# ==============================================================================
elif mode_idx == 3:
    st.subheader(
        tr(
            "Module 4: Deep Dive into HDBSCAN Hyperparameters (`eom` vs. `leaf` & `centroid` vs. `medoid`)",
            "Módulo 4: Hiperparámetros Clave de HDBSCAN (`eom` vs. `leaf` y `centroid` vs. `medoid`)",
        )
    )

    hp_tabs = st.tabs(
        [
            tr("1. `cluster_selection_method`: eom vs. leaf", "1. `cluster_selection_method`: eom vs. leaf"),
            tr("2. `store_centers`: centroid vs. medoid", "2. `store_centers`: centroid vs. medoid"),
        ]
    )

    # ---- Tab 1: cluster_selection_method (eom vs leaf) ----
    with hp_tabs[0]:
        st.info(
            tr(
                "**Compare `eom` (Excess of Mass) vs. `leaf`:** On hierarchical / multi-scale datasets where larger groups contain finer sub-clusters, **`eom`** maximizes total stability (preferring larger, persistent macro-clusters), while **`leaf`** always picks the finest clusters at the bottom of the condensed tree.",
                "**Compara `eom` (*Excess of Mass*) vs. `leaf`:** En datos jerárquicos o multiescala donde grupos grandes contienen subgrupos más finos, **`eom`** maximiza la estabilidad total (prefiriendo macro-clusters persistentes), mientras que **`leaf`** siempre elige los subgrupos más finos del fondo del árbol condensado.",
            )
        )

        ec1, ec2, ec3 = st.columns(3)
        mcs_hp = ec1.slider("min_cluster_size", 5, 40, 15, key="hp_mcs")
        ms_hp = ec2.slider("min_samples", 2, 20, 5, key="hp_ms")
        pat_hp = ec3.selectbox(
            tr("Dataset:", "Dataset:"),
            ["hierarchical", "blobs", "non_convex_mixed"],
            format_func=lambda k: {
                "hierarchical": tr("Multi-Scale Nested Clusters (Best for eom vs leaf)", "Clusters Anidados Multiescala (Ideal para eom vs leaf)"),
                "blobs": tr("Varying Densities (Dense + Sparse)", "Densidades Variables"),
                "non_convex_mixed": tr("Dense Moons + Sparse Cloud", "Lunas Densas + Nube Dispersa"),
            }[k],
            key="hp_pat",
        )

        X_hp, _ = make_varying_density_dataset(pat_hp, n_noise=20, seed=42)
        res_hp = cached_hdbscan_steps(X_hp, min_cluster_size=mcs_hp, min_samples=ms_hp)
        lbl_eom = res_hp["hdb_eom"].labels_
        lbl_leaf = res_hp["hdb_leaf"].labels_

        n_eom = len(set(lbl_eom) - {-1})
        n_leaf = len(set(lbl_leaf) - {-1})
        noise_eom = float(np.mean(lbl_eom == -1) * 100)
        noise_leaf = float(np.mean(lbl_leaf == -1) * 100)
        stab_eom = sum(res_hp["condensed"][c]["stability"] for c in res_hp["eom_selected"])
        stab_leaf = sum(res_hp["condensed"][c]["stability"] for c in res_hp["leaf_selected"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("EOM Clusters", f"{n_eom}", tr(f"Noise: {noise_eom:.1f}%", f"Ruido: {noise_eom:.1f}%"))
        m2.metric("Leaf Clusters", f"{n_leaf}", tr(f"Noise: {noise_leaf:.1f}%", f"Ruido: {noise_leaf:.1f}%"))
        m3.metric(tr("EOM Total Stability", "Estabilidad Total EOM"), f"{stab_eom:.1f}")
        m4.metric(tr("Leaf Total Stability", "Estabilidad Total Leaf"), f"{stab_leaf:.1f}")

        sc1, sc2 = st.columns(2)
        with sc1:
            fig_eom = px.scatter(
                x=X_hp[:, 0],
                y=X_hp[:, 1],
                color=format_cluster_labels(lbl_eom),
                color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#9CA3AF"},
                title=tr(f"`cluster_selection_method='eom'` → {n_eom} Clusters", f"`cluster_selection_method='eom'` → {n_eom} Clusters"),
                labels={"x": "$x_1$", "y": "$x_2$", "color": "EOM"},
            )
            fig_eom.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_eom.update_layout(height=400, template="plotly_white", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_eom, width="stretch")

            fig_t_eom = plot_condensed_tree(res_hp["condensed"], res_hp["eom_selected"], res_hp["leaf_selected"], "eom", tr)
            fig_t_eom.update_layout(height=360)
            st.plotly_chart(fig_t_eom, width="stretch")

        with sc2:
            fig_leaf = px.scatter(
                x=X_hp[:, 0],
                y=X_hp[:, 1],
                color=format_cluster_labels(lbl_leaf),
                color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#9CA3AF"},
                title=tr(f"`cluster_selection_method='leaf'` → {n_leaf} Clusters", f"`cluster_selection_method='leaf'` → {n_leaf} Clusters"),
                labels={"x": "$x_1$", "y": "$x_2$", "color": "Leaf"},
            )
            fig_leaf.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_leaf.update_layout(height=400, template="plotly_white", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_leaf, width="stretch")

            fig_t_leaf = plot_condensed_tree(res_hp["condensed"], res_hp["eom_selected"], res_hp["leaf_selected"], "leaf", tr)
            fig_t_leaf.update_layout(height=360)
            st.plotly_chart(fig_t_leaf, width="stretch")

        # Difference table
        if not ES:
            st.markdown(
                """
                #### Summary of Differences: `eom` vs. `leaf`
                | Aspect | **`eom` (Excess of Mass — Default)** | **`leaf`** |
                | :--- | :--- | :--- |
                | **Selection Rule** | Maximizes total cluster stability $\\sum S(C)$ across the condensed tree | Always selects the bottom leaf nodes of the condensed tree |
                | **Granularity** | **Coarser / Macro-clusters** (merges sub-clusters if parent is more stable) | **Finer / Micro-clusters** (keeps every distinct dense sub-group) |
                | **Noise Level** | Usually **lower % of noise** (bridges between sub-clusters stay inside parent) | Slightly **higher % of noise** (points between leaves fall out) |
                | **Cluster Size Balance** | Can produce 1 very large cluster + a few small ones | Tends to produce **more uniform, homogeneous** cluster sizes |
                | **Best Used When** | You want the most globally persistent, robust structure (general default) | You need fine-grained segmentation (e.g., niche customer micro-segments, fine topics) |
                """
            )
        else:
            st.markdown(
                """
                #### Resumen de Diferencias: `eom` vs. `leaf`
                | Aspecto | **`eom` (*Excess of Mass* — Por defecto)** | **`leaf`** |
                | :--- | :--- | :--- |
                | **Regla de Selección** | Maximiza la estabilidad total $\\sum S(C)$ a lo largo del árbol condensado | Selecciona siempre los nodos hoja del fondo del árbol condensado |
                | **Granularidad** | **Más gruesa / Macro-clusters** (fusiona subgrupos si el padre es más estable) | **Más fina / Micro-clusters** (conserva cada subgrupo denso separado) |
                | **Nivel de Ruido** | Suele generar **menos % de ruido** (puntos entre subgrupos quedan en el padre) | Ligeramente **más % de ruido** (los puntos entre hojas se descartan) |
                | **Equilibrio de Tamaños** | Puede crear un cluster muy grande junto a otros pequeños | Tiende a producir grupos **más homogéneos y parejos** en tamaño |
                | **Cuándo Usarlo** | Cuando buscas la estructura más persistente y robusta globalmente | Cuando necesitas segmentación fina (micro-segmentos de clientes, sub-temas) |
                """
            )

    # ---- Tab 2: store_centers (centroid vs medoid) ----
    with hp_tabs[1]:
        st.info(
            tr(
                "**Compare `store_centers='centroid'` vs. `store_centers='medoid'`:** On non-convex shapes (rings, crescent moons, C-shapes), the **centroid** (weighted average) can fall **in empty space outside the cluster** — or even inside a *different* cluster (try Concentric Rings!). The **medoid** is always an **actual observation** belonging to that cluster.",
                "**Compara `store_centers='centroid'` vs. `store_centers='medoid'`:** En formas no convexas (anillos, medias lunas, forma C), el **centroide** (promedio ponderado) puede caer **en espacio vacío fuera del cluster** —¡o incluso dentro de *otro* cluster (prueba Anillos Concéntricos)! El **medoide** siempre es una **observación real** que pertenece al cluster.",
            )
        )

        cc1, cc2 = st.columns([1.05, 2.0])
        with cc1:
            shape_c = st.selectbox(
                tr("Non-Convex Geometry:", "Geometría No Convexa:"),
                ["rings", "moons", "c_l"],
                format_func=lambda s: {
                    "rings": tr("Concentric Rings (Outer Centroid Falls Inside Inner Ring!)", "Anillos Concéntricos (¡El Centroide Exterior Cae en el Centro!)"),
                    "moons": tr("Interlocking Crescent Moons", "Medias Lunas Entrelazadas"),
                    "c_l": tr("C-Shape & L-Shape Manifolds", "Formas en C y en L"),
                }[s],
            )
            mcs_c = st.slider("min_cluster_size", 5, 35, 12, key="cent_mcs")
            ms_c = st.slider("min_samples", 2, 15, 5, key="cent_ms")

            X_c, _ = make_nonconvex_centers_dataset(shape_c, seed=42)
            hdb_c = HDBSCAN(
                min_cluster_size=mcs_c,
                min_samples=ms_c,
                store_centers="both",
                copy=True,
            ).fit(X_c)

            fig_cent, cent_rows = plot_centers_comparison(
                X_c,
                hdb_c.labels_,
                hdb_c.centroids_,
                hdb_c.medoids_,
                tr("HDBSCAN `store_centers='both'`: Centroids (★) vs. Medoids (◆)", "HDBSCAN `store_centers='both'`: Centroides (★) vs. Medoides (◆)"),
                tr,
            )

            if cent_rows:
                df_cent = pd.DataFrame(cent_rows).rename(
                    columns={
                        "Centroid Gap": tr("Centroid Gap to Real Data", "Brecha Centroide→Dato Real"),
                        "Medoid Gap": tr("Medoid Gap to Real Data", "Brecha Medoide→Dato Real"),
                    }
                )
                st.markdown(tr("#### Center Coordinates & Empty-Space Gap", "#### Coordenadas de Centros y Brecha al Vacío"))
                st.dataframe(df_cent, hide_index=True, use_container_width=True)
                max_gap = max(r["Centroid Gap"] for r in cent_rows)
                if max_gap > 0.25:
                    st.warning(
                        tr(
                            f"⚠️ Notice that a **Centroid** landed `{max_gap:.2f}` units away from any real point in its own cluster (in empty space!), whereas every **Medoid** has gap = `0.000` because it IS a real data point.",
                            f"⚠️ Observa que un **Centroide** cayó a `{max_gap:.2f}` unidades de cualquier punto real de su propio cluster (¡en el vacío!), mientras que cada **Medoide** tiene brecha = `0.000` porque ES un dato real.",
                        )
                    )

        with cc2:
            st.plotly_chart(fig_cent, width="stretch")

        if not ES:
            st.markdown(
                """
                #### Summary of Differences: `store_centers='centroid'` vs. `'medoid'`
                | Aspect | **`centroid`** | **`medoid`** |
                | :--- | :--- | :--- |
                | **Mathematical Definition** | Probability-weighted arithmetic mean: $\\mu_C = \\frac{\\sum_{i \\in C} p_i x_i}{\\sum_{i \\in C} p_i}$ | Actual point $x^* \\in C$ minimizing $\\sum_{j \\in C} d(x^*, x_j)$ |
                | **Guaranteed to Be a Real Data Point?** | **No** (synthetic average vector) | **Yes** (always an actual observation from your dataset) |
                | **Behavior on Non-Convex Shapes** | **Can land in empty space** or inside another cluster (e.g., concentric rings) | **Always stays on the cluster's real manifold** |
                | **Sensitivity to Outliers / Skew** | Moderate (skewed tails pull the mean) | **High robustness** (resistant to extreme values) |
                | **Computational Cost** | Very fast — $O(|C| \\cdot d)$ | Higher — $O(|C|^2)$ pairwise distances inside each cluster |
                | **Best Real-World Use Cases** | Fast Euclidean summaries, spherical clusters, downstream vector math | **Representative exemplars** (real customer persona, real patient record, real GPS hub, real image) |
                """
            )
        else:
            st.markdown(
                """
                #### Resumen de Diferencias: `store_centers='centroid'` vs. `'medoid'`
                | Aspecto | **`centroid`** | **`medoid`** |
                | :--- | :--- | :--- |
                | **Definición Matemática** | Media aritmética ponderada por probabilidad: $\\mu_C = \\frac{\\sum_{i \\in C} p_i x_i}{\\sum_{i \\in C} p_i}$ | Punto real $x^* \\in C$ que minimiza $\\sum_{j \\in C} d(x^*, x_j)$ |
                | **¿Es un Dato Real del Dataset?** | **No** (es un vector promedio sintético) | **Sí** (siempre es una observación real de tus datos) |
                | **En Formas No Convexas** | **Puede caer en espacio vacío** o dentro de otro cluster (ej. anillos concéntricos) | **Siempre permanece sobre la geometría real del cluster** |
                | **Sensibilidad a Asimetrías / Outliers** | Moderada (las colas arrastran el promedio) | **Alta robustez** (resistente a valores extremos) |
                | **Costo Computacional** | Muy rápido — $O(|C| \\cdot d)$ | Mayor — $O(|C|^2)$ distancias por pares dentro de cada cluster |
                | **Mejores Casos de Uso** | Resúmenes rápidos en espacio euclidiano, clusters esféricos | **Ejemplares representativos** (cliente real tipo, paciente representativo, hub GPS real, imagen real) |
                """
            )


# ==============================================================================
# MODULE 5: HDBSCAN VS. DBSCAN (VARIABLE DENSITIES & MASTER COMPARISON)
# ==============================================================================
elif mode_idx == 4:
    st.subheader(
        tr(
            "Module 5: HDBSCAN vs. DBSCAN — Solving the Variable-Density Dilemma",
            "Módulo 5: HDBSCAN vs. DBSCAN — Resolviendo el Dilema de las Densidades Variables",
        )
    )
    st.info(
        tr(
            "**Try to beat HDBSCAN with DBSCAN!** Move DBSCAN's single global radius **$\\varepsilon$** in the sidebar. With a small $\\varepsilon$, DBSCAN separates the dense clusters on the left but destroys the sparse cluster on the right into noise; with a large $\\varepsilon$, it captures the sparse cluster but merges the dense ones! **HDBSCAN needs no $\\varepsilon$** and separates all densities simultaneously.",
            "**¡Intenta ganarle a HDBSCAN usando DBSCAN!** Mueve el radio global único **$\\varepsilon$** de DBSCAN en la barra lateral. Con $\\varepsilon$ pequeño, DBSCAN separa los grupos densos de la izquierda pero destruye el grupo disperso de la derecha como ruido; con $\\varepsilon$ grande, captura el disperso pero fusiona los densos. **HDBSCAN no necesita $\\varepsilon$** y adapta la densidad localmente.",
        )
    )

    st.sidebar.header(tr("Showdown Controls", "Controles del Duelo"))
    pat_vs = st.sidebar.selectbox(
        tr("Variable-Density Scenario:", "Escenario de Densidad Variable:"),
        ["blobs", "non_convex_mixed", "hierarchical"],
        format_func=lambda k: {
            "blobs": tr("2 Tight Dense Blobs + 1 Wide Sparse Blob", "2 Grupos Muy Densos + 1 Grupo Amplio y Disperso"),
            "non_convex_mixed": tr("Dense Interlocking Moons + Sparse Cloud", "Medias Lunas Densas + Nube Dispersa"),
            "hierarchical": tr("Multi-Scale Hierarchical Clusters", "Clusters Jerárquicos Multiescala"),
        }[k],
    )
    eps_vs = st.sidebar.slider("DBSCAN eps (ε)", 0.10, 1.40, 0.35, step=0.05)
    min_db_vs = st.sidebar.slider("DBSCAN min_samples", 3, 20, 6)
    mcs_vs = st.sidebar.slider("HDBSCAN min_cluster_size", 5, 40, 15)
    ms_vs = st.sidebar.slider("HDBSCAN min_samples", 2, 20, 5)

    X_vs, y_vs = make_varying_density_dataset(pat_vs, n_noise=20, seed=42)
    lbl_db = DBSCAN(eps=eps_vs, min_samples=min_db_vs).fit_predict(X_vs)
    lbl_hdb = HDBSCAN(min_cluster_size=mcs_vs, min_samples=ms_vs, copy=True).fit_predict(X_vs)

    # Evaluate on non-noise ground truth points
    valid_gt = y_vs != -1
    ari_db = adjusted_rand_score(y_vs[valid_gt], lbl_db[valid_gt])
    ari_hdb = adjusted_rand_score(y_vs[valid_gt], lbl_hdb[valid_gt])
    nmi_db = normalized_mutual_info_score(y_vs[valid_gt], lbl_db[valid_gt])
    nmi_hdb = normalized_mutual_info_score(y_vs[valid_gt], lbl_hdb[valid_gt])

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("DBSCAN Clusters", f"{len(set(lbl_db) - {-1})}", f"Noise: {np.mean(lbl_db == -1)*100:.1f}%")
    v2.metric("DBSCAN ARI / NMI", f"{ari_db:.3f} / {nmi_db:.3f}")
    v3.metric("HDBSCAN Clusters", f"{len(set(lbl_hdb) - {-1})}", f"Noise: {np.mean(lbl_hdb == -1)*100:.1f}%")
    v4.metric("HDBSCAN ARI / NMI", f"{ari_hdb:.3f} / {nmi_hdb:.3f}")

    p_db, p_hdb = st.columns(2)
    with p_db:
        fig_db = px.scatter(
            x=X_vs[:, 0],
            y=X_vs[:, 1],
            color=format_cluster_labels(lbl_db),
            color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#9CA3AF"},
            title=tr(f"DBSCAN (Fixed ε={eps_vs:.2f}, min_samples={min_db_vs})", f"DBSCAN (ε fijo = {eps_vs:.2f}, min_samples={min_db_vs})"),
            labels={"x": "$x_1$", "y": "$x_2$", "color": "DBSCAN"},
        )
        fig_db.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_db.update_layout(height=440, template="plotly_white", legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig_db, width="stretch")

    with p_hdb:
        fig_hdb = px.scatter(
            x=X_vs[:, 0],
            y=X_vs[:, 1],
            color=format_cluster_labels(lbl_hdb),
            color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#9CA3AF"},
            title=tr(f"HDBSCAN (No ε! min_cluster_size={mcs_vs}, min_samples={ms_vs})", f"HDBSCAN (¡Sin ε! min_cluster_size={mcs_vs}, min_samples={ms_vs})"),
            labels={"x": "$x_1$", "y": "$x_2$", "color": "HDBSCAN"},
        )
        fig_hdb.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_hdb.update_layout(height=440, template="plotly_white", legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig_hdb, width="stretch")

    st.markdown("---")
    if not ES:
        st.markdown(
            """
            ### Master Comparison Table: DBSCAN vs. HDBSCAN
            | Criterion | **DBSCAN** | **HDBSCAN** |
            | :--- | :--- | :--- |
            | **Density Threshold (`eps`)** | Requires a **single fixed global $\\varepsilon$** | **No $\\varepsilon$ required** (evaluates all $\\varepsilon$ scales hierarchically) |
            | **Varying Cluster Densities** | **Struggles** (either shatters sparse clusters or merges dense ones) | **Excellent** (extracts clusters at different density levels) |
            | **Key Hyperparameters** | `eps` ($\\varepsilon$, hard to tune in high-D) and `min_samples` | `min_cluster_size` (intuitive!), `min_samples`, `cluster_selection_method` |
            | **Underlying Structure** | Flat $\\varepsilon$-neighborhood graph | **Mutual Reachability MST → Condensed Tree → Cluster Stability** |
            | **Output Information** | Hard labels only (`0, 1, ..., -1`) | Labels + **soft membership probabilities (`probabilities_`)** + optional **`centroids_` / `medoids_`** |
            | **Granularity Control** | Must re-run with different $\\varepsilon$ | Switch seamlessly between **`eom`** (macro) and **`leaf`** (micro) |
            | **Speed / Complexity** | Faster on flat queries: $O(N \\log N)$ with spatial index | Slightly slower due to MST + tree condensation: $O(N \\log N)$ to $O(N^2)$ |
            """
        )
    else:
        st.markdown(
            """
            ### Tabla Comparativa Maestra: DBSCAN vs. HDBSCAN
            | Criterio | **DBSCAN** | **HDBSCAN** |
            | :--- | :--- | :--- |
            | **Umbral de Densidad (`eps`)** | Exige un **único radio global fijo $\\varepsilon$** | **No requiere $\\varepsilon$** (explora todas las escalas jerárquicamente) |
            | **Clusters de Densidad Variable** | **Sufre mucho** (destruye grupos dispersos o fusiona los densos) | **Excelente** (extrae grupos en distintos niveles de densidad) |
            | **Hiperparámetros Clave** | `eps` ($\\varepsilon$, difícil de calibrar en alta dimensión) y `min_samples` | `min_cluster_size` (intuitivo), `min_samples`, `cluster_selection_method` |
            | **Estructura Matemática** | Grafo plano de vecindarios-$\\varepsilon$ | **MST de Alcanzabilidad Mutua → Árbol Condensado → Estabilidad** |
            | **Información de Salida** | Solo etiquetas rígidas (`0, 1, ..., -1`) | Etiquetas + **probabilidades suaves (`probabilities_`)** + **`centroids_` / `medoids_`** opcionales |
            | **Control de Granularidad** | Hay que re-ejecutar cambiando $\\varepsilon$ | Cambia entre **`eom`** (macro-grupos) y **`leaf`** (micro-grupos) |
            | **Velocidad / Complejidad** | Más rápido en consultas planas: $O(N \\log N)$ con índice espacial | Ligeramente más costoso por MST y condensación: $O(N \\log N)$ a $O(N^2)$ |
            """
        )


# ==============================================================================
# MODULE 6: REAL-WORLD CASE STUDIES & AUTOENCODER + HDBSCAN PIPELINE
# ==============================================================================
elif mode_idx == 5:
    st.subheader(
        tr(
            "Module 6: Real-World Case Studies & The Autoencoder + HDBSCAN Pipeline",
            "Módulo 6: Casos del Mundo Real y el Pipeline Autoencoder + HDBSCAN",
        )
    )

    rw_tabs = st.tabs(
        [
            tr("1. Fraud / Anomaly Detection (Autoencoder)", "1. Detección de Fraude / Anomalías (Autoencoder)"),
            tr("2. Image Denoising (Autoencoder)", "2. Eliminación de Ruido en Imágenes (Autoencoder)"),
            tr("3. Urban & Rural GPS Hotspots (HDBSCAN)", "3. Hotspots GPS Urbanos y Rurales (HDBSCAN)"),
            tr("4. Deep Clustering: Autoencoder + HDBSCAN", "4. Clustering Profundo: Autoencoder + HDBSCAN"),
        ]
    )

    # ---- Case 1: Fraud / Industrial Anomaly Detection via Autoencoder ----
    with rw_tabs[0]:
        st.markdown(
            tr(
                "**Real-World Scenario (Banking Fraud & Predictive Maintenance):** Train an Autoencoder **exclusively on normal transactions/telemetry**. Because it only learns the normal manifold, when an anomalous transaction arrives, the Autoencoder **fails to reconstruct it accurately** $\\to$ a spike in Reconstruction MSE flags the anomaly!",
                "**Escenario Real (Fraude Bancario y Mantenimiento Predictivo):** Entrenamos un Autoencoder **exclusivamente con transacciones/telemetría normales**. Como solo aprende la estructura de lo normal, cuando llega una transacción anómala el Autoencoder **no logra reconstruirla bien** $\\to$ un pico en el Error de Reconstrucción (MSE) dispara la alerta.",
            )
        )
        rng_f = np.random.default_rng(7)
        # Normal 8D telemetry living on a 2D nonlinear manifold (correlated spending/sensor patterns)
        n_norm = 360
        t_u = rng_f.uniform(-2.0, 2.0, n_norm)
        t_v = rng_f.uniform(-1.5, 1.5, n_norm)
        X_norm = np.column_stack(
            [
                t_u,
                t_v,
                np.sin(t_u) + 0.3 * t_v,
                np.cos(t_v) - 0.2 * t_u,
                0.5 * t_u**2 - 0.4 * t_v,
                t_u * t_v * 0.4,
                0.7 * t_u + 0.5 * t_v,
                -0.6 * t_u + 0.8 * np.sin(t_v),
            ]
        ) + rng_f.normal(0, 0.08, (n_norm, 8))

        n_anom = 28
        X_anom = rng_f.uniform(-2.6, 2.6, (n_anom, 8))
        scaler_f = StandardScaler().fit(X_norm)
        X_norm_s = scaler_f.transform(X_norm)
        X_anom_s = scaler_f.transform(X_anom)
        X_all_f = np.vstack([X_norm_s, X_anom_s])
        y_true_f = np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_anom, dtype=int)])

        ae_fraud = train_cached_autoencoder(X_norm_s, None, hidden_dim=16, latent_dim=2, activation="tanh", epochs=200, lr=0.015, batch_size=64, seed=7)
        errs_all = ae_fraud.reconstruction_errors(X_all_f)
        errs_norm = errs_all[:n_norm]

        fc1, fc2 = st.columns([1.0, 2.0])
        with fc1:
            pct_thr = st.slider(
                tr("Anomaly Threshold (Percentile of Normal MSE):", "Umbral de Anomalía (Percentil del MSE Normal):"),
                85.0,
                99.5,
                96.0,
                step=0.5,
            )
            thr_val = float(np.percentile(errs_norm, pct_thr))
            y_pred_f = (errs_all > thr_val).astype(int)
            prec = precision_score(y_true_f, y_pred_f, zero_division=0)
            rec = recall_score(y_true_f, y_pred_f, zero_division=0)
            f1 = f1_score(y_true_f, y_pred_f, zero_division=0)

            st.metric(tr("MSE Threshold", "Umbral MSE"), f"{thr_val:.4f}")
            st.metric(tr("Precision (Alert Accuracy)", "Precisión (Exactitud de Alertas)"), f"{prec*100:.1f}%")
            st.metric(tr("Recall (Fraud Caught)", "Recall (Fraude Detectado)"), f"{rec*100:.1f}%")
            st.metric("F1-Score", f"{f1:.3f}")

        with fc2:
            df_err = pd.DataFrame(
                {
                    "MSE": errs_all,
                    "Type": [tr("Normal", "Normal")] * n_norm + [tr("Fraud / Anomaly", "Fraude / Anomalía")] * n_anom,
                }
            )
            fig_err = px.histogram(
                df_err,
                x="MSE",
                color="Type",
                barmode="overlay",
                nbins=45,
                opacity=0.72,
                color_discrete_map={
                    tr("Normal", "Normal"): "#3B82F6",
                    tr("Fraud / Anomaly", "Fraude / Anomalía"): "#EF4444",
                },
                title=tr("Reconstruction Error Distribution: Normal vs. Anomalies", "Distribución del Error de Reconstrucción: Normal vs. Anomalías"),
            )
            fig_err.add_vline(
                x=thr_val,
                line_dash="dash",
                line_color="#111827",
                annotation_text=tr(f"Threshold = {thr_val:.3f}", f"Umbral = {thr_val:.3f}"),
            )
            fig_err.update_layout(height=420, template="plotly_white")
            st.plotly_chart(fig_err, width="stretch")

    # ---- Case 2: Denoising Autoencoder on Handwritten Digits ----
    with rw_tabs[1]:
        st.markdown(
            tr(
                "**Real-World Scenario (Medical Imaging & Photo Restoration):** A **Denoising Autoencoder** receives *corrupted* images as input ($\\tilde{x} = x + \\epsilon$) but is trained to reconstruct the *clean* image $x$. The bottleneck forces the network to discard random pixel static and keep only coherent shapes!",
                "**Escenario Real (Imágenes Médicas y Restauración de Fotos):** Un **Autoencoder Eliminador de Ruido (*Denoising Autoencoder*)** recibe imágenes *corrompidas* ($\\tilde{x} = x + \\epsilon$) pero se entrena para reconstruir la imagen *limpia* $x$. El cuello de botella obliga a descartar la estática aleatoria y conservar solo las formas coherentes.",
            )
        )
        dc1, dc2 = st.columns(2)
        noise_lvl = dc1.slider(tr("Gaussian Noise Level (σ)", "Nivel de Ruido Gaussiano (σ)"), 0.10, 0.50, 0.28, step=0.03)
        img_pick = dc2.slider(tr("Test Digit Sample #", "Muestra de Dígito de Prueba #"), 0, 79, 5)

        dig_d = load_digits()
        X_clean = dig_d.data[:480] / 16.0
        rng_d = np.random.default_rng(123)
        X_noisy = np.clip(X_clean + rng_d.normal(0.0, noise_lvl, X_clean.shape), 0.0, 1.0)

        X_tr_clean, X_te_clean = X_clean[:400], X_clean[400:]
        X_tr_noisy, X_te_noisy = X_noisy[:400], X_noisy[400:]

        dae = train_cached_autoencoder(
            X_tr_noisy,
            X_tr_clean,
            hidden_dim=32,
            latent_dim=10,
            activation="tanh",
            epochs=180,
            lr=0.015,
            batch_size=64,
            seed=123,
        )
        X_te_denoised = np.clip(dae.reconstruct(X_te_noisy), 0.0, 1.0)

        mse_before = float(np.mean((X_te_noisy - X_te_clean) ** 2))
        mse_after = float(np.mean((X_te_denoised - X_te_clean) ** 2))
        dm1, dm2, dm3 = st.columns(3)
        dm1.metric(tr("Test MSE Before Denoising", "MSE de Prueba Antes"), f"{mse_before:.4f}")
        dm2.metric(tr("Test MSE After Denoising", "MSE de Prueba Después"), f"{mse_after:.4f}")
        dm3.metric(tr("Error Reduction", "Reducción del Error"), f"{(1 - mse_after / max(mse_before, 1e-9)) * 100:.1f}%")

        ic1, ic2, ic3 = st.columns(3)
        for col_img, arr, ttl in [
            (ic1, X_te_noisy[img_pick].reshape(8, 8), tr("1. Noisy Input (Corrupted)", "1. Entrada con Ruido")),
            (ic2, X_te_denoised[img_pick].reshape(8, 8), tr("2. Autoencoder Denoised Output", "2. Salida Limpia del Autoencoder")),
            (ic3, X_te_clean[img_pick].reshape(8, 8), tr("3. Ground Truth Clean Image", "3. Imagen Limpia Real")),
        ]:
            fig_im = px.imshow(arr, color_continuous_scale="gray_r", zmin=0, zmax=1, title=ttl)
            fig_im.update_coloraxes(showscale=False)
            fig_im.update_xaxes(showticklabels=False)
            fig_im.update_yaxes(showticklabels=False)
            fig_im.update_layout(height=300)
            col_img.plotly_chart(fig_im, width="stretch")

    # ---- Case 3: Urban & Rural GPS Hotspots (HDBSCAN + Medoids) ----
    with rw_tabs[2]:
        st.markdown(
            tr(
                "**Real-World Scenario (Logistics & Mobility):** Delivery GPS pings form **ultra-dense downtown hotspots** alongside **wide, sparse suburban/rural hubs**, plus stray highway pings (noise). HDBSCAN discovers both urban and rural zones simultaneously, and `store_centers='medoid'` selects an **actual real drop-off location** as the optimal micro-hub!",
                "**Escenario Real (Logística y Movilidad):** Las coordenadas GPS de entregas forman **hotspots urbanos ultra-densos** junto a **zonas suburbanas/rurales amplias y dispersas**, además de puntos aislados en carretera (ruido). HDBSCAN descubre zonas urbanas y rurales simultáneamente, y `store_centers='medoid'` elige una **ubicación real existente** como centro de distribución óptimo.",
            )
        )
        rng_g = np.random.default_rng(99)
        # Dense downtown zones + sparse rural towns + stray GPS noise
        downtown_1 = rng_g.normal([-89.22, 13.70], [0.008, 0.006], (90, 2))
        downtown_2 = rng_g.normal([-89.18, 13.71], [0.007, 0.007], (85, 2))
        suburb = rng_g.normal([-89.08, 13.66], [0.032, 0.028], (75, 2))
        rural = rng_g.normal([-89.30, 13.60], [0.045, 0.038], (65, 2))
        stray = np.column_stack([rng_g.uniform(-89.38, -89.00, 25), rng_g.uniform(13.53, 13.76, 25)])
        X_gps = np.vstack([downtown_1, downtown_2, suburb, rural, stray])

        gc1, gc2 = st.columns([1.0, 2.0])
        with gc1:
            gps_method = st.radio(
                tr("HDBSCAN `cluster_selection_method`:", "HDBSCAN `cluster_selection_method`:"),
                ["eom", "leaf"],
                horizontal=True,
                key="gps_method",
            )
            gps_mcs = st.slider("min_cluster_size", 8, 35, 18, key="gps_mcs")
            hdb_gps = HDBSCAN(
                min_cluster_size=gps_mcs,
                min_samples=6,
                cluster_selection_method=gps_method,
                store_centers="medoid",
                copy=True,
            ).fit(X_gps)
            n_zones = len(set(hdb_gps.labels_) - {-1})
            st.metric(tr("Discovered Delivery Zones", "Zonas de Entrega Descubiertas"), f"{n_zones}")
            st.metric(tr("Filtered Stray GPS Pings (-1)", "Pings GPS Aislados Filtrados (-1)"), f"{int(np.sum(hdb_gps.labels_ == -1))}")
            st.success(
                tr(
                    "Green diamonds (◆) are the **Medoids** (`store_centers='medoid'`): each is an actual real GPS delivery coordinate that minimizes total travel distance within its zone!",
                    "Los diamantes verdes (◆) son los **Medoides** (`store_centers='medoid'`): ¡cada uno es una coordenada GPS real que minimiza la distancia total dentro de su zona!",
                )
            )
        with gc2:
            fig_gps = px.scatter(
                x=X_gps[:, 0],
                y=X_gps[:, 1],
                color=format_cluster_labels(hdb_gps.labels_),
                color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#9CA3AF"},
                title=tr("Urban & Rural GPS Hotspots Discovered by HDBSCAN + Medoid Hubs", "Hotspots GPS Urbanos y Rurales Descubiertos por HDBSCAN + Hubs Medoides"),
                labels={"x": tr("Longitude", "Longitud"), "y": tr("Latitude", "Latitud"), "color": tr("Zone", "Zona")},
            )
            if hdb_gps.medoids_ is not None and len(hdb_gps.medoids_) > 0:
                fig_gps.add_trace(
                    go.Scatter(
                        x=hdb_gps.medoids_[:, 0],
                        y=hdb_gps.medoids_[:, 1],
                        mode="markers",
                        name=tr("Medoid Hub (Real Location)", "Hub Medoide (Ubicación Real)"),
                        marker=dict(size=15, symbol="diamond", color="#10B981", line=dict(width=2, color="black")),
                    )
                )
            fig_gps.update_layout(height=450, template="plotly_white", legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig_gps, width="stretch")

    # ---- Case 4: Deep Clustering Synergy (Autoencoder + HDBSCAN) ----
    with rw_tabs[3]:
        st.markdown(
            tr(
                "**Why Combine Autoencoders + HDBSCAN?** In high-dimensional spaces (like 64D images or 30D medical records), distances suffer from the **Curse of Dimensionality**, causing density-based clustering to mark most points as noise (`-1`). Compressing the data first with an **Autoencoder** into a compact latent space makes **HDBSCAN** dramatically more accurate!",
                "**¿Por Qué Combinar Autoencoders + HDBSCAN?** En espacios de alta dimensión (como imágenes de 64D o registros médicos de 30D), las distancias sufren la **Maldición de la Dimensionalidad**, haciendo que el clustering por densidad marque casi todo como ruido (`-1`). ¡Comprimir primero los datos con un **Autoencoder** hacia un espacio latente compacto hace que **HDBSCAN** sea mucho más preciso!",
            )
        )
        pc1, pc2, pc3 = st.columns(3)
        ds_pipe = pc1.selectbox(
            tr("High-D Real Dataset:", "Dataset Real de Alta Dimensión:"),
            ["wine", "breast_cancer", "digits"],
            format_func=lambda k: {
                "wine": tr("Wine Chemistry (13D, 3 classes)", "Química de Vinos (13D, 3 clases)"),
                "breast_cancer": tr("Breast Cancer Wisconsin (30D, 2 classes)", "Cáncer de Mama Wisconsin (30D, 2 clases)"),
                "digits": tr("Handwritten Digits (64D, 10 classes)", "Dígitos Manuscritos (64D, 10 clases)"),
            }[k],
        )
        mcs_pipe = pc2.slider("HDBSCAN min_cluster_size", 5, 30, 12, key="pipe_mcs")
        sel_pipe = pc3.radio("cluster_selection_method", ["eom", "leaf"], horizontal=True, key="pipe_sel")

        X_hd, y_hd, _ = load_real_highd_dataset(ds_pipe, max_samples=420, seed=42)

        # 1. Raw High-D HDBSCAN
        lbl_raw = HDBSCAN(min_cluster_size=mcs_pipe, min_samples=5, cluster_selection_method=sel_pipe, copy=True).fit_predict(X_hd)
        # 2. PCA 2D + HDBSCAN
        Z_pca_hd = PCA(n_components=2, random_state=42).fit_transform(X_hd)
        lbl_pca_hd = HDBSCAN(min_cluster_size=mcs_pipe, min_samples=5, cluster_selection_method=sel_pipe, copy=True).fit_predict(Z_pca_hd)
        # 3. Autoencoder 2D + HDBSCAN
        ae_hd = train_cached_autoencoder(X_hd, None, hidden_dim=24, latent_dim=2, activation="tanh", epochs=220, lr=0.012, batch_size=64, seed=42)
        Z_ae_hd = ae_hd.encode(X_hd)
        lbl_ae_hd = HDBSCAN(min_cluster_size=mcs_pipe, min_samples=5, cluster_selection_method=sel_pipe, copy=True).fit_predict(Z_ae_hd)

        def eval_pipe(lbls):
            return (
                len(set(lbls) - {-1}),
                float(np.mean(lbls == -1) * 100),
                float(adjusted_rand_score(y_hd, lbls)),
                float(normalized_mutual_info_score(y_hd, lbls)),
            )

        r_raw, r_pca, r_ae = eval_pipe(lbl_raw), eval_pipe(lbl_pca_hd), eval_pipe(lbl_ae_hd)
        df_pipe = pd.DataFrame(
            [
                {
                    tr("Pipeline", "Pipeline"): tr(f"Raw {X_hd.shape[1]}D → HDBSCAN", f"Crudo {X_hd.shape[1]}D → HDBSCAN"),
                    "Clusters": r_raw[0],
                    tr("Noise %", "% Ruido"): f"{r_raw[1]:.1f}%",
                    "ARI": round(r_raw[2], 3),
                    "NMI": round(r_raw[3], 3),
                },
                {
                    tr("Pipeline", "Pipeline"): "PCA (2D) → HDBSCAN",
                    "Clusters": r_pca[0],
                    tr("Noise %", "% Ruido"): f"{r_pca[1]:.1f}%",
                    "ARI": round(r_pca[2], 3),
                    "NMI": round(r_pca[3], 3),
                },
                {
                    tr("Pipeline", "Pipeline"): "Autoencoder (2D) → HDBSCAN",
                    "Clusters": r_ae[0],
                    tr("Noise %", "% Ruido"): f"{r_ae[1]:.1f}%",
                    "ARI": round(r_ae[2], 3),
                    "NMI": round(r_ae[3], 3),
                },
            ]
        )
        st.dataframe(df_pipe, hide_index=True, use_container_width=True)

        pp1, pp2 = st.columns(2)
        with pp1:
            fig_pae = px.scatter(
                x=Z_ae_hd[:, 0],
                y=Z_ae_hd[:, 1],
                color=format_cluster_labels(lbl_ae_hd),
                color_discrete_map={tr("Noise (-1)", "Ruido (-1)"): "#9CA3AF"},
                title=tr("Autoencoder 2D Latent Space Clustered by HDBSCAN", "Espacio Latente 2D del Autoencoder Agrupado por HDBSCAN"),
                labels={"x": "z₁", "y": "z₂", "color": "HDBSCAN"},
            )
            fig_pae.update_layout(height=420, template="plotly_white", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_pae, width="stretch")

        with pp2:
            fig_gt = px.scatter(
                x=Z_ae_hd[:, 0],
                y=Z_ae_hd[:, 1],
                color=y_hd.astype(str),
                title=tr("Ground Truth Labels in Autoencoder 2D Latent Space", "Etiquetas Reales en el Espacio Latente 2D del Autoencoder"),
                labels={"x": "z₁", "y": "z₂", "color": tr("True Class", "Clase Real")},
            )
            fig_gt.update_layout(height=420, template="plotly_white", legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_gt, width="stretch")
