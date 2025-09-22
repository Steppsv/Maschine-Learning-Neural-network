import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def initialize_centroids(X, K, rng=None):
    """Случайно берём K пикселей как стартовые центроиды."""
    H, W, _ = X.shape
    if rng is None:
        rng = np.random.default_rng()
    hs = rng.integers(0, H, size=K)
    ws = rng.integers(0, W, size=K)
    return X[hs, ws].astype(np.float32)  # (K,3)


def assign_clusters_vec(X, centroids):
    """
    Векторно: расстояния от каждого пикселя до всех центроидов (H,W,K),
    берём индекс минимума → метка кластера (H,W).
    """
    dists = np.linalg.norm(X[..., None, :] - centroids[None, None, :, :], axis=3)
    labels = np.argmin(dists, axis=2)  # (H,W)
    return labels


def update_centroids_vec(X, labels, K, rng=None):
    """
    Векторно считаем сумму компонент и количество элементов в каждом кластере.
    Пустые кластеры реинициализируем случайными пикселями.
    """
    H, W, _ = X.shape
    Xf = X.reshape(-1, 3)              # (N,3)
    lab = labels.reshape(-1)           # (N,)
    counts = np.bincount(lab, minlength=K).astype(np.float64)  # (K,)

    # суммы по каждому каналу
    sums = np.vstack([
        np.bincount(lab, weights=Xf[:, c], minlength=K)
        for c in range(3)
    ]).T  # (K,3)

    centroids = sums / np.maximum(counts[:, None], 1.0)        # защита от деления на 0
    empty = (counts == 0)
    if np.any(empty):
        if rng is None:
            rng = np.random.default_rng()
        rnd_idx = rng.integers(0, Xf.shape[0], size=int(empty.sum()))
        centroids[empty] = Xf[rnd_idx]

    return centroids.astype(np.float32)


def compute_sse(X, labels, centroids):
    """Векторно считаем SSE (сумма квадратов расстояний до центроида)."""
    recon = centroids[labels]          # (H,W,3)
    diff = X - recon
    return float(np.sum(diff * diff))


def kmeans_image(
    X,
    K=2,
    tol=1e-6,
    max_iter=300,
    random_state=None,
    track_sse=True,
):
    """
    Быстрый K-Means для изображения.
    Останов по ||C_new - C_old||_F < tol.
    Возвращает (labels, centroids, sse_trace).
    """
    rng = np.random.default_rng(random_state)
    centroids = initialize_centroids(X, K, rng=rng)
    sse_trace = []

    for _ in range(max_iter):
        labels = assign_clusters_vec(X, centroids)
        new_centroids = update_centroids_vec(X, labels, K, rng=rng)

        # критерий останова по сдвигу центроидов
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if track_sse:
            sse_trace.append(compute_sse(X, labels, centroids))

        if shift < tol:
            break

    return labels, centroids, sse_trace


def main():
    # === Параметры ===
    image_path = "car.jpg"  # положи файл рядом со скриптом
    K = 8                 # число цветов/кластеров
    random_state = 42       # для воспроизводимости (можно None)
    tol = 1e-6
    max_iter = 300

    # === Загрузка изображения и нормализация ===
    with Image.open(image_path) as img:
        X = np.array(img, dtype=np.float32) / 255.0  # (H,W,3) в [0,1]

    # === K-Means ===
    labels, centroids, sse_trace = kmeans_image(
        X, K=K, tol=tol, max_iter=max_iter, random_state=random_state, track_sse=True
    )

    # === Реконструкция/квантование цветов ===
    Y = centroids[labels]  # (H,W,3)

    # === Визуализация ===
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(X)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Quantized (K={K})")
    plt.imshow(Y)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    if len(sse_trace) > 0:
        plt.figure()
        plt.plot(sse_trace, marker='o')
        plt.title("SSE per iteration")
        plt.xlabel("Iteration")
        plt.ylabel("SSE")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()