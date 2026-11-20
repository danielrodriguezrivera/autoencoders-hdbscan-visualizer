import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
    make_s_curve,
    make_swiss_roll,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# PURE-NUMPY AUTOENCODER (FAST, TRANSPARENT, DETERMINISTIC)
# ==============================================================================
class NumpyAutoencoder:
    """
    Lightweight 4-layer MLP Autoencoder (Input -> Hidden -> Latent -> Hidden -> Input)
    trained with Adam optimizer in pure NumPy.

    Supports 'tanh', 'relu', and 'linear' hidden activations so users can directly
    observe why a linear autoencoder spans the PCA subspace while nonlinear activations
    unroll curved manifolds.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        latent_dim: int = 2,
        activation: str = "tanh",
        lr: float = 0.015,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.activation = activation
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.layer_dims = [self.input_dim, self.hidden_dim, self.latent_dim, self.hidden_dim, self.input_dim]

        rng = np.random.default_rng(self.seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for d_in, d_out in zip(self.layer_dims[:-1], self.layer_dims[1:]):
            # Xavier / He-like initialization scaled for stability
            scale = np.sqrt(2.0 / (d_in + d_out)) if activation != "relu" else np.sqrt(2.0 / d_in)
            self.weights.append(rng.normal(0.0, scale, size=(d_in, d_out)))
            self.biases.append(np.zeros((1, d_out), dtype=float))

        self.loss_history: list[float] = []

    def _act(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "relu":
            return np.maximum(0.0, x)
        return x  # linear

    def _act_deriv(self, pre_act: np.ndarray, post_act: np.ndarray) -> np.ndarray:
        if self.activation == "tanh":
            return 1.0 - post_act**2
        if self.activation == "relu":
            return (pre_act > 0.0).astype(float)
        return np.ones_like(pre_act)

    def _forward_full(self, X: np.ndarray):
        # Layer 0: Input -> Hidden1 (activated)
        z1 = X @ self.weights[0] + self.biases[0]
        a1 = self._act(z1)
        # Layer 1: Hidden1 -> Latent (linear bottleneck for unconstrained latent coords)
        z_lat = a1 @ self.weights[1] + self.biases[1]
        # Layer 2: Latent -> Hidden2 (activated)
        z3 = z_lat @ self.weights[2] + self.biases[2]
        a3 = self._act(z3)
        # Layer 3: Hidden2 -> Output (linear reconstruction)
        x_hat = a3 @ self.weights[3] + self.biases[3]
        return z1, a1, z_lat, z3, a3, x_hat

    def encode(self, X: np.ndarray) -> np.ndarray:
        a1 = self._act(X @ self.weights[0] + self.biases[0])
        return a1 @ self.weights[1] + self.biases[1]

    def decode(self, Z: np.ndarray) -> np.ndarray:
        a3 = self._act(Z @ self.weights[2] + self.biases[2])
        return a3 @ self.weights[3] + self.biases[3]

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        return self._forward_full(X)[-1]

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2, axis=1)

    def fit(
        self,
        X_in: np.ndarray,
        X_target: np.ndarray | None = None,
        epochs: int = 250,
        batch_size: int | None = None,
    ) -> "NumpyAutoencoder":
        if X_target is None:
            X_target = X_in
        n = len(X_in)
        rng = np.random.default_rng(self.seed + 1)

        m_w = [np.zeros_like(w) for w in self.weights]
        v_w = [np.zeros_like(w) for w in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        self.loss_history = []

        use_batch = batch_size is not None and 0 < batch_size < n

        for _ in range(int(epochs)):
            if use_batch:
                perm = rng.permutation(n)
                batches = [perm[i : i + batch_size] for i in range(0, n, batch_size)]
            else:
                batches = [np.arange(n)]

            for idx in batches:
                xb = X_in[idx]
                yb = X_target[idx]
                m = len(xb)
                z1, a1, z_lat, z3, a3, x_hat = self._forward_full(xb)

                # MSE gradient: d(mean((x_hat - y)^2)) / d(x_hat)
                d_out = (2.0 / (m * self.input_dim)) * (x_hat - yb)

                # Layer 3 grads
                gw3 = a3.T @ d_out + self.weight_decay * self.weights[3]
                gb3 = np.sum(d_out, axis=0, keepdims=True)

                # Layer 2 grads
                da3 = d_out @ self.weights[3].T
                dz3 = da3 * self._act_deriv(z3, a3)
                gw2 = z_lat.T @ dz3 + self.weight_decay * self.weights[2]
                gb2 = np.sum(dz3, axis=0, keepdims=True)

                # Layer 1 grads (linear bottleneck)
                dz_lat = dz3 @ self.weights[2].T
                gw1 = a1.T @ dz_lat + self.weight_decay * self.weights[1]
                gb1 = np.sum(dz_lat, axis=0, keepdims=True)

                # Layer 0 grads
                da1 = dz_lat @ self.weights[1].T
                dz1 = da1 * self._act_deriv(z1, a1)
                gw0 = xb.T @ dz1 + self.weight_decay * self.weights[0]
                gb0 = np.sum(dz1, axis=0, keepdims=True)

                grads_w = [gw0, gw1, gw2, gw3]
                grads_b = [gb0, gb1, gb2, gb3]

                step += 1
                bc1 = 1.0 - beta1**step
                bc2 = 1.0 - beta2**step
                for i in range(4):
                    m_w[i] = beta1 * m_w[i] + (1.0 - beta1) * grads_w[i]
                    v_w[i] = beta2 * v_w[i] + (1.0 - beta2) * (grads_w[i] ** 2)
                    self.weights[i] -= self.lr * (m_w[i] / bc1) / (np.sqrt(v_w[i] / bc2) + eps)

                    m_b[i] = beta1 * m_b[i] + (1.0 - beta1) * grads_b[i]
                    v_b[i] = beta2 * v_b[i] + (1.0 - beta2) * (grads_b[i] ** 2)
                    self.biases[i] -= self.lr * (m_b[i] / bc1) / (np.sqrt(v_b[i] / bc2) + eps)

            full_hat = self.reconstruct(X_in)
            self.loss_history.append(float(np.mean((full_hat - X_target) ** 2)))

        return self


# ==============================================================================
# HDBSCAN STEP-BY-STEP ENGINE (CORE DIST, MUTUAL REACHABILITY, MST, CONDENSED TREE, STABILITY)
# ==============================================================================
def compute_hdbscan_steps(X: np.ndarray, min_cluster_size: int = 10, min_samples: int = 5) -> dict:
    """
    Explicitly compute every internal stage of HDBSCAN for educational visualization:
      1. Core distances core_k(x)
      2. Mutual reachability distance matrix D_mreach
      3. Minimum Spanning Tree (MST) edges sorted by weight
      4. Single-linkage hierarchy -> Condensed Tree with min_cluster_size
      5. Exact Cluster Stability S(C) = sum_{p in C} (lambda_p - lambda_birth(C))
      6. EOM (Excess of Mass) vs. Leaf selection on the exact condensed tree
    Also runs sklearn.cluster.HDBSCAN (eom & leaf, with store_centers='both') for verification.
    """
    n = len(X)
    k = min(max(1, int(min_samples)), n - 1)
    mcs = min(max(2, int(min_cluster_size)), max(2, n // 2))

    # 1. Core distances
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    dists_nn, _ = nn.kneighbors(X)
    core_dists = dists_nn[:, -1]

    # 2. Mutual reachability distance matrix
    euclid = squareform(pdist(X, metric="euclidean"))
    mreach = np.maximum(euclid, np.maximum(core_dists[:, None], core_dists[None, :]))
    np.fill_diagonal(mreach, 0.0)

    # 3. Minimum Spanning Tree (MST)
    mst_sparse = minimum_spanning_tree(mreach)
    mst_coo = mst_sparse.tocoo()
    mst_edges = [
        (int(u), int(v), float(w))
        for u, v, w in zip(mst_coo.row, mst_coo.col, mst_coo.data)
    ]
    mst_edges.sort(key=lambda e: e[2])

    # 4. Build single-linkage binary tree bottom-up via Union-Find
    parent_uf = list(range(2 * n - 1))
    size_uf = [1] * n + [0] * (n - 1)

    def find(a: int) -> int:
        while parent_uf[a] != a:
            parent_uf[a] = parent_uf[parent_uf[a]]
            a = parent_uf[a]
        return a

    # Node structure for single-linkage tree: node_id -> (left, right, weight, size)
    sl_children: dict[int, tuple[int, int, float, int]] = {}
    sl_members: dict[int, list[int]] = {i: [i] for i in range(n)}
    next_id = n
    for u, v, w in mst_edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        new_size = size_uf[ru] + size_uf[rv]
        sl_children[next_id] = (ru, rv, w, new_size)
        sl_members[next_id] = sl_members[ru] + sl_members[rv]
        parent_uf[ru] = next_id
        parent_uf[rv] = next_id
        size_uf[next_id] = new_size
        next_id += 1

    root_sl = next_id - 1

    # 5. Condense the single-linkage tree top-down using min_cluster_size
    # Each condensed cluster stores:
    #   id, parent, children, lambda_birth, lambda_death, initial_size,
    #   stability, subtree_stability, is_leaf, events [(lambda_val, remaining_size)]
    condensed: dict[int, dict] = {}
    cid_counter = 0

    def new_cluster(parent_cid: int | None, lam_birth: float, init_size: int) -> int:
        nonlocal cid_counter
        cid = cid_counter
        cid_counter += 1
        condensed[cid] = {
            "id": cid,
            "parent": parent_cid,
            "children": [],
            "lambda_birth": float(lam_birth),
            "lambda_death": float(lam_birth),
            "initial_size": int(init_size),
            "stability": 0.0,
            "subtree_stability": 0.0,
            "is_leaf": True,
            "events": [(float(lam_birth), int(init_size))],
            "members": [],
        }
        return cid

    root_cid = new_cluster(None, 0.0, n)
    condensed[root_cid]["members"] = list(range(n))

    # Queue of (sl_node_id, condensed_cluster_id)
    stack = [(root_sl, root_cid)]
    while stack:
        sl_node, cid = stack.pop()
        if sl_node < n:
            # Single point reached
            continue
        left, right, w, total_sz = sl_children[sl_node]
        lam = 1.0 / max(w, 1e-6)
        sz_l = size_uf[left]
        sz_r = size_uf[right]
        c_info = condensed[cid]

        if sz_l >= mcs and sz_r >= mcs:
            # True cluster split! Parent ends here; all remaining points contribute (lam - lam_birth)
            c_info["stability"] += total_sz * (lam - c_info["lambda_birth"])
            c_info["lambda_death"] = lam
            c_info["events"].append((lam, total_sz))
            c_info["is_leaf"] = False

            cid_l = new_cluster(cid, lam, sz_l)
            cid_r = new_cluster(cid, lam, sz_r)
            condensed[cid_l]["members"] = sl_members[left]
            condensed[cid_r]["members"] = sl_members[right]
            c_info["children"] = [cid_l, cid_r]
            stack.append((left, cid_l))
            stack.append((right, cid_r))
        elif sz_l < mcs and sz_r < mcs:
            # Both branches too small -> all points fall out of cid at lambda
            c_info["stability"] += total_sz * (lam - c_info["lambda_birth"])
            c_info["lambda_death"] = max(c_info["lambda_death"], lam)
            c_info["events"].append((lam, 0))
        else:
            # One branch falls out as noise, the larger branch continues as cid
            if sz_l >= mcs:
                big_node, small_sz, big_sz = left, sz_r, sz_l
            else:
                big_node, small_sz, big_sz = right, sz_l, sz_r
            c_info["stability"] += small_sz * (lam - c_info["lambda_birth"])
            c_info["lambda_death"] = max(c_info["lambda_death"], lam)
            c_info["events"].append((lam, big_sz))
            stack.append((big_node, cid))

    # 6. Compute EOM and Leaf selections on our condensed tree
    leaf_selected = {cid for cid, info in condensed.items() if info["is_leaf"] and cid != root_cid}

    # Bottom-up EOM pass (children always have larger cid than parents, so reverse cid order works)
    eom_selected: set[int] = set()
    eom_subtrees: dict[int, set[int]] = {}
    for cid in sorted(condensed.keys(), reverse=True):
        info = condensed[cid]
        if info["is_leaf"]:
            info["subtree_stability"] = info["stability"]
            eom_subtrees[cid] = {cid}
        else:
            c1, c2 = info["children"]
            children_stab = condensed[c1]["subtree_stability"] + condensed[c2]["subtree_stability"]
            if cid != root_cid and info["stability"] >= children_stab:
                info["subtree_stability"] = info["stability"]
                eom_subtrees[cid] = {cid}
            else:
                info["subtree_stability"] = children_stab
                eom_subtrees[cid] = eom_subtrees[c1] | eom_subtrees[c2]

    if root_cid in eom_subtrees and len(condensed) > 1:
        eom_selected = eom_subtrees[root_cid] - {root_cid}
    else:
        eom_selected = set()

    # Run official scikit-learn HDBSCAN for exact labels, probabilities, centroids & medoids
    hdb_eom = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=k,
        cluster_selection_method="eom",
        store_centers="both",
        copy=True,
    ).fit(X)
    hdb_leaf = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=k,
        cluster_selection_method="leaf",
        store_centers="both",
        copy=True,
    ).fit(X)

    return {
        "core_dists": core_dists,
        "euclid": euclid,
        "mreach": mreach,
        "mst_edges": mst_edges,
        "condensed": condensed,
        "eom_selected": eom_selected,
        "leaf_selected": leaf_selected,
        "hdb_eom": hdb_eom,
        "hdb_leaf": hdb_leaf,
    }


def mst_connected_components(n: int, mst_edges: list[tuple[int, int, float]], threshold: float, min_size: int = 1) -> np.ndarray:
    """Cut all MST edges with weight > threshold and return cluster labels (-1 if component < min_size)."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v, w in mst_edges:
        if w <= threshold:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

    comps: dict[int, list[int]] = {}
    for i in range(n):
        r = find(i)
        comps.setdefault(r, []).append(i)

    labels = np.full(n, -1, dtype=int)
    cid = 0
    for members in sorted(comps.values(), key=len, reverse=True):
        if len(members) >= min_size:
            labels[members] = cid
            cid += 1
    return labels


# ==============================================================================
# SYNTHETIC & REAL-WORLD DATASET GENERATORS
# ==============================================================================
def make_3d_manifold(kind: str, n_samples: int = 360, noise: float = 0.08, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate classic 3D nonlinear manifolds for Autoencoder vs. PCA comparison."""
    rng = np.random.default_rng(int(seed))
    if kind == "swiss_roll":
        X, color = make_swiss_roll(n_samples=n_samples, noise=noise * 2.5, random_state=int(seed))
    elif kind == "s_curve":
        X, color = make_s_curve(n_samples=n_samples, noise=noise, random_state=int(seed))
    else:  # 3d_helix_clusters: two intertwined 3D helices + central core
        n_each = n_samples // 3
        t1 = np.linspace(0, 3.2 * np.pi, n_each)
        h1 = np.column_stack([np.cos(t1), np.sin(t1), t1 / 3.0]) + rng.normal(0, noise, (n_each, 3))
        h2 = np.column_stack([-np.cos(t1), -np.sin(t1), t1 / 3.0]) + rng.normal(0, noise, (n_each, 3))
        n_rem = n_samples - 2 * n_each
        h3 = rng.normal([0.0, 0.0, 1.6], [0.22 + noise, 0.22 + noise, 0.85], (n_rem, 3))
        X = np.vstack([h1, h2, h3])
        color = np.concatenate([np.zeros(n_each), np.ones(n_each), np.full(n_rem, 2.0)])
    X = StandardScaler().fit_transform(X)
    return X, color


def make_varying_density_dataset(pattern: str = "blobs", n_noise: int = 25, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D datasets with dramatically different cluster densities + background noise."""
    rng = np.random.default_rng(int(seed))
    if pattern == "blobs":
        # Two tight dense clusters on the left + one wide sparse cluster on the right
        X, y = make_blobs(
            n_samples=[95, 95, 90],
            centers=[[-2.2, 0.85], [-2.2, -0.85], [2.4, 0.0]],
            cluster_std=[0.20, 0.20, 1.05],
            random_state=int(seed),
        )
    elif pattern == "hierarchical":
        # Multi-scale nested clusters: ideal for showcasing EOM vs. Leaf!
        # Left macro-group = 2 close dense sub-clusters; Right macro-group = 3 sub-clusters; Bottom = 1 loose cluster
        X, y = make_blobs(
            n_samples=[55, 55, 50, 50, 50, 65],
            centers=[
                [-2.4, 1.35],
                [-1.75, 0.75],
                [1.6, 1.45],
                [2.35, 1.15],
                [1.95, 0.45],
                [0.0, -2.0],
            ],
            cluster_std=[0.18, 0.18, 0.17, 0.17, 0.17, 0.55],
            random_state=int(seed),
        )
    else:  # non_convex_mixed: dense moons + sparse blob
        X_m, y_m = make_moons(n_samples=180, noise=0.055, random_state=int(seed))
        X_m = X_m * 1.35 + np.array([-1.6, 0.3])
        X_b = rng.normal([2.2, 0.2], [0.78, 0.78], size=(85, 2))
        X = np.vstack([X_m, X_b])
        y = np.concatenate([y_m, np.full(85, 2, dtype=int)])

    if n_noise > 0:
        mins = X.min(axis=0) - 0.5
        maxs = X.max(axis=0) + 0.5
        noise_pts = rng.uniform(mins, maxs, size=(int(n_noise), 2))
        X = np.vstack([X, noise_pts])
        y = np.concatenate([y, np.full(int(n_noise), -1, dtype=int)])

    return X, y


def make_nonconvex_centers_dataset(shape: str = "rings", seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate non-convex shapes where centroids fall in empty space while medoids stay on the manifold."""
    rng = np.random.default_rng(int(seed))
    if shape == "rings":
        X, y = make_circles(n_samples=300, factor=0.38, noise=0.05, random_state=int(seed))
        X = StandardScaler().fit_transform(X)
    elif shape == "moons":
        X, y = make_moons(n_samples=280, noise=0.06, random_state=int(seed))
        X = StandardScaler().fit_transform(X)
    else:  # c_shape_and_l_shape
        theta = np.linspace(-0.78 * np.pi, 0.78 * np.pi, 160)
        c_pts = np.column_stack([1.4 * np.cos(theta) - 1.2, 1.4 * np.sin(theta)]) + rng.normal(0, 0.07, (160, 2))
        l_horiz = np.column_stack([np.linspace(0.6, 2.4, 70), np.full(70, -1.0)]) + rng.normal(0, 0.07, (70, 2))
        l_vert = np.column_stack([np.full(70, 0.6), np.linspace(-1.0, 1.2, 70)]) + rng.normal(0, 0.07, (70, 2))
        X = np.vstack([c_pts, l_horiz, l_vert])
        y = np.concatenate([np.zeros(160, dtype=int), np.ones(140, dtype=int)])
        X = StandardScaler().fit_transform(X)

    # Add a few isolated outliers
    outliers = rng.uniform(X.min(axis=0) - 0.4, X.max(axis=0) + 0.4, size=(12, 2))
    X = np.vstack([X, outliers])
    y = np.concatenate([y, np.full(12, -1, dtype=int)])
    return X, y


def load_real_highd_dataset(name: str, max_samples: int = 450, seed: int = 42):
    """Load standardized high-dimensional real datasets for Autoencoder + HDBSCAN pipeline."""
    rng = np.random.default_rng(int(seed))
    if name == "digits":
        data = load_digits()
        # Keep 5 visually distinct digits (0, 1, 3, 4, 7) or all 10? Let's keep 5 clean digits or subsample all 10
        X, y = data.data, data.target
        if len(X) > max_samples:
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X, y = X[idx], y[idx]
        target_names = [str(i) for i in range(10)]
    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        target_names = list(data.target_names)
    else:  # breast_cancer
        data = load_breast_cancer()
        X, y = data.data, data.target
        if len(X) > max_samples:
            idx = rng.choice(len(X), size=max_samples, replace=False)
            X, y = X[idx], y[idx]
        target_names = list(data.target_names)

    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y, target_names
