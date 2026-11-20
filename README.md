# Autoencoders & HDBSCAN: Theory, Math & Interactive Visualizations

<p align="center">
  <a href="#-english">🇬🇧 English</a> •
  <a href="#-español">🇪🇸 Español</a>
</p>

---

## 🇬🇧 English

An interactive bilingual (**English / Spanish**) web application built with **Python, Streamlit, Plotly, NumPy, SciPy, and Scikit-Learn** that visually explains **Autoencoders** and **HDBSCAN** (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*) from first principles—combining intuitive analogies, essential mathematics, real-world case studies, and interactive visualizations.

### Author

**Daniel Edgardo Rodríguez Rivera**
* Location: El Salvador
* LinkedIn: [https://www.linkedin.com/in/daniel-rodriguez-sv/](https://www.linkedin.com/in/daniel-rodriguez-sv/)
* GitHub: [https://github.com/danielrodriguezrivera/](https://github.com/danielrodriguezrivera/)

### Key Features & Interactive Modules

* **Bilingual Interface (English & Spanish):** Instant language toggle in the sidebar (`Español` / `English`).
* **Essential Theory & Intuitive Analogies:**
  * **Autoencoders:** *The "Sketch Artist & Sculptor" Analogy* explaining bottlenecks, compression, and reconstruction loss.
  * **HDBSCAN:** *The "Archipelago Rising from the Sea" Analogy* explaining variable-density hierarchies, condensed trees, and cluster stability.
* **1. Autoencoders — Encoder, Latent Space & Decoder:**
  * Dynamic neural network architecture diagram ($x \to h_1 \to z \to h_2 \to \hat{x}$) updating live with hidden units and activation choice.
  * Interactive guide comparing **Hidden Layer Units** (underfitting vs. capacity vs. overfitting) and **Activation Functions** (`tanh`, `relu`, `linear`) with plotted activation curves.
  * Live **MSE training curve** and an interactive **2D Latent Space Explorer** where moving $(z_1, z_2)$ displays both the **nearest real original digit** and the **Decoder's synthesized output** side by side.
* **2. Dimensionality Reduction — Autoencoder vs. PCA:**
  * Side-by-side comparison on nonlinear 3D manifolds (**Swiss Roll**, **S-Curve**, **Intertwined 3D Helices**) and 64D **Handwritten Digits**.
  * Switch the Autoencoder activation to `linear` to verify that a linear Autoencoder spans the same subspace as **PCA**, while nonlinear activations (`tanh`, `relu`) bend to fit curved manifolds.
* **3. HDBSCAN Step-by-Step — MST & Cluster Stability:**
  * **Step 1:** Core Distance $\text{core}_k(a)$ and Mutual Reachability Distance $d_{\text{mreach-}k}(a, b)$ with interactive per-point radius inspection.
  * **Step 2:** Interactive **Minimum Spanning Tree (MST)** with a live edge-weight threshold slider to prune the heaviest bridges and watch clusters emerge.
  * **Step 3:** Interactive **Condensed Tree** where each shaded funnel's area visualizes **Cluster Stability** $S(C) = \sum_{p \in C} (\lambda_p - \lambda_{\text{birth}}(C))$, plus a node-by-node stability table and bar chart.
  * **Step 4:** Final HDBSCAN clusters and soft membership probabilities (`probabilities_`).
* **4. HDBSCAN Hyperparameters (`cluster_selection_method` & `store_centers`):**
  * **`cluster_selection_method` (`eom` vs. `leaf`):** Side-by-side comparison on multi-scale nested clusters with highlighted condensed trees, total stability metrics, and a practical decision guide.
  * **`store_centers` (`centroid` vs. `medoid`):** Interactive non-convex geometries (concentric rings, interlocking moons, C/L shapes) exposing when **centroids** land in empty space while **medoids** remain actual observations on the data manifold.
* **5. HDBSCAN vs. DBSCAN:**
  * Variable-density showdown illustrating DBSCAN's single-$\varepsilon$ dilemma vs. HDBSCAN's multi-scale density adaptation, with ARI/NMI metrics and a master comparison table.
* **6. Real-World Case Studies & Autoencoder + HDBSCAN Pipeline:**
  * **Fraud / Anomaly Detection (Autoencoder):** Flagging anomalies via Reconstruction MSE threshold with Precision, Recall, and F1-Score.
  * **Image Denoising (Denoising Autoencoder):** Restoring noisy 8×8 handwritten digits.
  * **Urban & Rural GPS Hotspots (HDBSCAN + Medoids):** Variable-density delivery zone discovery using real medoid micro-hubs.
  * **Deep Clustering Pipeline (Autoencoder + HDBSCAN):** Benchmarking Raw High-D HDBSCAN vs. PCA + HDBSCAN vs. Autoencoder + HDBSCAN on `Wine`, `Breast Cancer`, and `Handwritten Digits`.

### Project Structure

```text
autoencoders-hdbscan-visualizer/
├── app.py              # Main Streamlit application & interactive modules
├── models.py           # Pure-NumPy Autoencoder, HDBSCAN step-by-step engine & datasets
├── viz.py              # Reusable Plotly visualizations (architecture, MST, condensed tree, centers)
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT License
└── README.md           # Bilingual documentation (EN / ES)
```

### How to Run Locally

1. Clone or navigate to the repository:
   ```bash
   git clone https://github.com/danielrodriguezrivera/autoencoders-hdbscan-visualizer.git
   cd autoencoders-hdbscan-visualizer
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Open your browser at `http://localhost:8501`.

---

## 🇪🇸 Español

Una aplicación web interactiva y bilingüe (**Español / Inglés**) construida con **Python, Streamlit, Plotly, NumPy, SciPy y Scikit-Learn** que explica visualmente los **Autoencoders** y **HDBSCAN** (*Hierarchical Density-Based Spatial Clustering of Applications with Noise*) desde sus fundamentos, combinando analogías intuitivas, teoría matemática esencial, casos del mundo real y visualizaciones interactivas.

### Autor

**Daniel Edgardo Rodríguez Rivera**
* Ubicación: El Salvador
* LinkedIn: [https://www.linkedin.com/in/daniel-rodriguez-sv/](https://www.linkedin.com/in/daniel-rodriguez-sv/)
* GitHub: [https://github.com/danielrodriguezrivera/](https://github.com/danielrodriguezrivera/)

### Características y Módulos Interactivos

* **Interfaz Bilingüe (Español e Inglés):** Cambio instantáneo de idioma desde la barra lateral (`Español` / `English`).
* **Teoría Esencial y Analogías Intuitivas:**
  * **Autoencoders:** *La Analogía del "Caricaturista y el Escultor"* para comprender el cuello de botella, la compresión y el error de reconstrucción.
  * **HDBSCAN:** *La Analogía del "Archipiélago que Emerge del Mar"* para entender jerarquías de densidad variable, árboles condensados y estabilidad de clusters.
* **1. Autoencoders — Codificador, Espacio Latente y Decodificador:**
  * Diagrama dinámico de arquitectura neuronal ($x \to h_1 \to z \to h_2 \to \hat{x}$) que se actualiza en vivo según el número de neuronas ocultas y la función de activación.
  * Guía interactiva sobre **Neuronas en Capa Oculta** (*underfitting* vs. capacidad vs. *overfitting*) y diferencias entre **Funciones de Activación** (`tanh`, `relu`, `linear`) con gráfico comparativo de curvas.
  * Curva de pérdida **MSE** en vivo y **Explorador del Espacio Latente 2D** donde al mover $(z_1, z_2)$ se muestra tanto la **entrada original real más cercana** como la **imagen sintetizada por el Decodificador**.
* **2. Reducción de Dimensionalidad — Autoencoder vs. PCA:**
  * Comparativa lado a lado en variedades no lineales 3D (**Rollo Suizo**, **Curva S**, **Hélices 3D Entrelazadas**) y **Dígitos Manuscritos** (64D $\to$ 2D).
  * Permite cambiar la activación del Autoencoder a `linear` para comprobar que un Autoencoder lineal recupera el mismo subespacio que **PCA**, mientras que las activaciones no lineales (`tanh`, `relu`) se adaptan a geometrías curvas.
* **3. HDBSCAN Paso a Paso — MST y Estabilidad de Clusters:**
  * **Paso 1:** Distancia Núcleo $\text{core}_k(a)$ y Distancia de Alcanzabilidad Mutua $d_{\text{mreach-}k}(a, b)$ con inspector interactivo por punto.
  * **Paso 2:** **Árbol de Expansión Mínima (MST)** interactivo con deslizador de umbral para cortar las aristas más pesadas en vivo y observar cómo emergen los clusters.
  * **Paso 3:** **Árbol Condensado** interactivo donde el área sombreada de cada rama representa la **Estabilidad de Clusters** $S(C) = \sum_{p \in C} (\lambda_p - \lambda_{\text{birth}}(C))$, junto a una tabla y gráfico de estabilidad por nodo.
  * **Paso 4:** Clusters finales de HDBSCAN y probabilidades suaves de pertenencia (`probabilities_`).
* **4. Hiperparámetros de HDBSCAN (`cluster_selection_method` y `store_centers`):**
  * **`cluster_selection_method` (`eom` vs. `leaf`):** Comparativa lado a lado en clusters jerárquicos multiescala con árboles condensados resaltados, estabilidad total y tabla de decisión.
  * **`store_centers` (`centroid` vs. `medoid`):** Geometrías no convexas (anillos concéntricos, medias lunas, formas en C/L) que muestran cómo un **centroide** puede caer en espacio vacío mientras que el **medoide** siempre es un punto real del dataset.
* **5. HDBSCAN vs. DBSCAN:**
  * Duelo interactivo en escenarios de densidades variables que expone el dilema del radio global $\varepsilon$ en DBSCAN frente a la adaptación multiescala de HDBSCAN, con métricas ARI/NMI y tabla comparativa maestra.
* **6. Casos del Mundo Real y Pipeline Autoencoder + HDBSCAN:**
  * **Detección de Fraude / Anomalías (Autoencoder):** Detección mediante umbral de error de reconstrucción (MSE) con Precisión, Recall y F1-Score.
  * **Eliminación de Ruido en Imágenes (*Denoising Autoencoder*):** Restauración de dígitos manuscritos 8×8 con ruido gaussiano.
  * **Hotspots GPS Urbanos y Rurales (HDBSCAN + Medoides):** Descubrimiento simultáneo de zonas urbanas densas y rurales dispersas seleccionando ubicaciones reales como hubs medoides.
  * **Pipeline de Clustering Profundo (Autoencoder + HDBSCAN):** Comparativa entre HDBSCAN en datos crudos de alta dimensión vs. PCA + HDBSCAN vs. Autoencoder + HDBSCAN sobre `Wine`, `Breast Cancer` y `Handwritten Digits`.

### Estructura del Proyecto

```text
autoencoders-hdbscan-visualizer/
├── app.py              # Aplicación principal de Streamlit y módulos interactivos
├── models.py           # Autoencoder en NumPy puro, motor paso a paso de HDBSCAN y datasets
├── viz.py              # Visualizaciones reutilizables en Plotly (arquitectura, MST, árbol condensado, centros)
├── requirements.txt    # Dependencias de Python
├── LICENSE             # Licencia MIT
└── README.md           # Documentación bilingüe (EN / ES)
```

### Cómo Ejecutar Localmente

1. Clona o entra al directorio del repositorio:
   ```bash
   git clone https://github.com/danielrodriguezrivera/autoencoders-hdbscan-visualizer.git
   cd autoencoders-hdbscan-visualizer
   ```
2. Instala las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta la aplicación de Streamlit:
   ```bash
   streamlit run app.py
   ```
4. Abre tu navegador en `http://localhost:8501`.

---

## License / Licencia

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
