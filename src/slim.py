"""
Iteración 3 - SLIM y FISM

SLIM:   min  ||X – XW||²_F  +  (λ_A/2)||W||²_F  +  λ_B||W||_1
         s.t. diag(W) = 0,  W ≥ 0

FISM:   W = PQᵀ  (P, Q ∈ ℝ^{n_items × f})
         min  ||X – X(PQᵀ)||²_F  +  (λ_A/2)(||P||²_F + ||Q||²_F)  +  λ_B(||P||_1 + ||Q||_1)

La pérdida usa reduce_sum (no reduce_mean) para que el gradiente de reconstrucción
sea comparable a los términos de regularización.

Dataset: data/trimmed_dataset.zip (4347 playlists, 1704 tracks únicos, 29 playlists de test)

Uso:
    python src/slim.py --mode slim  --epochs 500 --lr 0.01 --lambda-a 5.0 --lambda-b 1.0
    python src/slim.py --mode fism  --epochs 2000 --lr 0.001 --lambda-a 1.0 --lambda-b 0.0 --factors 64
"""

import argparse
import csv
import json
import logging
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Set

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEAM_NAME  = "Jacobo_Cousillas_Xaime_Paz_Nicolas_Aller"
TEAM_EMAIL = "jacobo.cousillas@udc.es_xaime.paz.ollero@udc.es_nicolas.aller@udc.es"
RECOMMENDATIONS_COUNT = 500


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_trimmed_data(zip_path: Path):
    """Lee train_trimmed.json y construye la matriz de interacciones."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("train_trimmed.json") as f:
            data = json.loads(f.read())

    track_counter: Counter = Counter()
    track_to_idx: dict = {}
    playlist_to_idx: dict = {}
    rows, cols = [], []

    for playlist in data["playlists"]:
        pid = str(playlist["pid"])
        if pid not in playlist_to_idx:
            playlist_to_idx[pid] = len(playlist_to_idx)
        p_idx = playlist_to_idx[pid]
        for track in playlist.get("tracks", []):
            uri = track["track_uri"]
            track_counter[uri] += 1
            if uri not in track_to_idx:
                track_to_idx[uri] = len(track_to_idx)
            rows.append(p_idx)
            cols.append(track_to_idx[uri])

    n_users = len(playlist_to_idx)
    n_items = len(track_to_idx)
    vals = np.ones(len(rows), dtype=np.float32)
    X = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    popular_tracks = [uri for uri, _ in track_counter.most_common()]
    idx_to_track = {v: k for k, v in track_to_idx.items()}

    density = X.nnz / (n_users * n_items) * 100
    logging.info(
        f"Train: {n_users} playlists, {n_items} tracks, {X.nnz} interacciones "
        f"(densidad {density:.3f}%)"
    )
    return X, track_to_idx, idx_to_track, popular_tracks


def load_test_data(zip_path: Path):
    """Lee test_input_trimmed.json y test_eval_trimmed.json."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("test_input_trimmed.json") as f:
            test_input = json.loads(f.read())["playlists"]
        with zf.open("test_eval_trimmed.json") as f:
            test_eval = json.loads(f.read())["playlists"]
    return test_input, test_eval


# ---------------------------------------------------------------------------
# Utilidades de recomendación (mismo patrón que knn.py / puresvd.py)
# ---------------------------------------------------------------------------

def build_seed_indices(seed_uris: Set[str], track_to_idx: dict) -> list:
    return [track_to_idx[uri] for uri in seed_uris if uri in track_to_idx]


def top500_from_scores(scores: np.ndarray, seed_indices: list, idx_to_track: dict) -> list:
    scores[seed_indices] = 0.0
    nonzero_indices = np.where(scores > 0)[0]
    if len(nonzero_indices) == 0:
        return []
    nonzero_scores = scores[nonzero_indices]
    if len(nonzero_indices) <= RECOMMENDATIONS_COUNT:
        order = np.argsort(-nonzero_scores)
    else:
        order = np.argpartition(-nonzero_scores, RECOMMENDATIONS_COUNT)[:RECOMMENDATIONS_COUNT]
        order = order[np.argsort(-nonzero_scores[order])]
    return [idx_to_track[nonzero_indices[i]] for i in order if nonzero_scores[i] > 0]



def popularity_fallback(popular_tracks: list, exclude_uris: Set[str], n: int) -> list:
    result = []
    for uri in popular_tracks:
        if uri not in exclude_uris:
            result.append(uri)
            if len(result) == n:
                break
    return result


# ---------------------------------------------------------------------------
# Entrenamiento SLIM
# ---------------------------------------------------------------------------

def train_slim(
    X_tf: tf.Tensor,
    n_items: int,
    epochs: int,
    lr: float,
    lambda_a: float,
    lambda_b: float,
) -> np.ndarray:
    """
    Entrena SLIM con Adam y GradientTape. Aplica las restricciones diag(W)=0 y W≥0
    como proyección después de cada actualización de gradiente.
    Devuelve W como numpy array.
    """
    tf.random.set_seed(42)
    W = tf.Variable(tf.random.uniform((n_items, n_items), 0.0, 0.01, seed=42), dtype=tf.float32)
    diag_mask = tf.constant(1.0 - np.eye(n_items, dtype=np.float32))
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            XW = X_tf @ W
            loss_rec = tf.reduce_sum(tf.square(X_tf - XW))
            reg_l2 = (lambda_a / 2.0) * tf.reduce_sum(tf.square(W))
            reg_l1 = lambda_b * tf.reduce_sum(tf.abs(W))
            loss = loss_rec + reg_l2 + reg_l1
        grad = tape.gradient(loss, W)
        optimizer.apply_gradients([(grad, W)])
        return loss

    logging.info(
        f"Entrenando SLIM: {n_items} items, {epochs} epochs, "
        f"lr={lr}, λ_A={lambda_a}, λ_B={lambda_b}"
    )
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_step()
        # Proyección a las restricciones: diag=0 y no-negatividad
        W.assign(tf.maximum(W, 0.0) * diag_mask)
        if epoch % 50 == 0:
            logging.info(f"  Epoch {epoch:4d}/{epochs}  loss={loss.numpy():.6f}  ({time.time()-t0:.1f}s)")

    logging.info(f"Entrenamiento SLIM completado en {time.time()-t0:.1f}s")
    W_np = W.numpy()
    sparsity = np.mean(W_np == 0) * 100
    logging.info(f"  Sparsidad de W: {sparsity:.1f}%  (max={W_np.max():.4f})")
    return W_np


# ---------------------------------------------------------------------------
# Entrenamiento FISM
# ---------------------------------------------------------------------------

def train_fism(
    X_tf: tf.Tensor,
    n_items: int,
    factors: int,
    epochs: int,
    lr: float,
    lambda_a: float,
    lambda_b: float,
):
    """
    Entrena FISM (S = PQᵀ) con Adam y GradientTape.
    Devuelve (P, Q) como numpy arrays.
    """
    tf.random.set_seed(42)
    P = tf.Variable(tf.random.normal((n_items, factors), stddev=0.01, seed=42), dtype=tf.float32)
    Q = tf.Variable(tf.random.normal((n_items, factors), stddev=0.01, seed=43), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            XP     = X_tf @ P                       # (n_users, f)
            scores = XP @ tf.transpose(Q)           # (n_users, n_items)
            loss_rec = tf.reduce_sum(tf.square(X_tf - scores))
            reg_l2 = (lambda_a / 2.0) * (
                tf.reduce_sum(tf.square(P)) + tf.reduce_sum(tf.square(Q))
            )
            reg_l1 = lambda_b * (
                tf.reduce_sum(tf.abs(P)) + tf.reduce_sum(tf.abs(Q))
            )
            loss = loss_rec + reg_l2 + reg_l1
        grads = tape.gradient(loss, [P, Q])
        optimizer.apply_gradients(zip(grads, [P, Q]))
        return loss

    logging.info(
        f"Entrenando FISM: {n_items} items, f={factors}, {epochs} epochs, "
        f"lr={lr}, λ_A={lambda_a}, λ_B={lambda_b}"
    )
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_step()
        if epoch % 50 == 0:
            logging.info(f"  Epoch {epoch:4d}/{epochs}  loss={loss.numpy():.6f}  ({time.time()-t0:.1f}s)")

    logging.info(f"Entrenamiento FISM completado en {time.time()-t0:.1f}s")
    return P.numpy(), Q.numpy()


# ---------------------------------------------------------------------------
# Evaluación inline
# ---------------------------------------------------------------------------

def _compute_metrics(recs: list, ground_truth: Set[str]) -> tuple:
    n_gt = len(ground_truth)
    if n_gt == 0:
        return 0.0, 0.0, 51.0
    relevant_top = [t for t in recs[:n_gt] if t in ground_truth]
    r_prec = len(relevant_top) / n_gt
    dcg_val = sum(1.0 / np.log2(i + 2) for i, t in enumerate(recs) if t in ground_truth)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(n_gt))
    ndcg_val = dcg_val / ideal_dcg if ideal_dcg > 0 else 0.0
    clicks_val = next((i // 10 for i, t in enumerate(recs) if t in ground_truth), 51.0)
    return r_prec, ndcg_val, float(clicks_val)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def generate_slim(
    mode: str,
    epochs: int,
    lr: float,
    lambda_a: float,
    lambda_b: float,
    factors: int,
) -> str:
    BASE_DIR       = Path(__file__).resolve().parent.parent
    ZIP_PATH       = BASE_DIR / "data" / "trimmed_dataset.zip"
    SUBMISSIONS    = BASE_DIR / "submissions"
    SUBMISSIONS.mkdir(exist_ok=True)

    # --- Datos ---
    X, track_to_idx, idx_to_track, popular_tracks = load_trimmed_data(ZIP_PATH)
    test_input, test_eval = load_test_data(ZIP_PATH)
    n_items = X.shape[1]

    ground_truth = {
        str(pl["pid"]): {t["track_uri"] for t in pl.get("tracks", [])}
        for pl in test_eval
    }

    # Tensor denso de entrenamiento (~30 MB para 4347×1704)
    logging.info("Convirtiendo matriz a tensor denso TF...")
    X_tf = tf.constant(X.toarray(), dtype=tf.float32)

    # --- Entrenamiento ---
    if mode == "slim":
        W = train_slim(X_tf, n_items, epochs, lr, lambda_a, lambda_b)
        param_str   = f"e{epochs}_lr{lr}_la{lambda_a}_lb{lambda_b}"
        output_name = f"iteracion_3_slim_{param_str}.csv"
    else:
        P, Q = train_fism(X_tf, n_items, factors, epochs, lr, lambda_a, lambda_b)
        param_str   = f"f{factors}_e{epochs}_lr{lr}_la{lambda_a}_lb{lambda_b}"
        output_name = f"iteracion_3_fism_{param_str}.csv"

    # --- Recomendaciones ---
    logging.info("Generando recomendaciones para playlists de test...")
    rows_out = []
    r_precs, ndcgs, clicks_list = [], [], []
    fallback_count = 0

    for playlist in test_input:
        pid        = str(playlist["pid"])
        seed_uris  = {t["track_uri"] for t in playlist.get("tracks", [])}
        seed_idx   = build_seed_indices(seed_uris, track_to_idx)

        if seed_idx:
            x_u = np.zeros(n_items, dtype=np.float32)
            x_u[seed_idx] = 1.0
            if mode == "slim":
                scores = x_u @ W                    # (n_items,)
            else:
                scores = (x_u @ P) @ Q.T            # (n_items,)
        else:
            scores = np.zeros(n_items, dtype=np.float32)

        recs = top500_from_scores(scores.copy(), list(seed_idx), idx_to_track)

        if len(recs) < RECOMMENDATIONS_COUNT:
            fallback_count += 1
            exclude = seed_uris | set(recs)
            recs += popularity_fallback(popular_tracks, exclude, RECOMMENDATIONS_COUNT - len(recs))

        rows_out.append((pid, recs))

        if pid in ground_truth:
            rp, nd, cl = _compute_metrics(recs, ground_truth[pid])
            r_precs.append(rp)
            ndcgs.append(nd)
            clicks_list.append(cl)

    logging.info(f"Playlists con fallback a popularidad: {fallback_count}/{len(test_input)}")

    # --- Escritura CSV ---
    output_path = SUBMISSIONS / output_name
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["team_info", TEAM_NAME, TEAM_EMAIL])
        for pid, recs in rows_out:
            assert len(recs) == RECOMMENDATIONS_COUNT, f"PID {pid}: {len(recs)} recs"
            writer.writerow([pid] + recs)
    logging.info(f"Submission escrita: {output_path}")

    # --- Métricas ---
    meta: dict = {
        "mode":             mode,
        "epochs":           epochs,
        "lr":               lr,
        "lambda_a":         lambda_a,
        "lambda_b":         lambda_b,
        "factors":          factors if mode == "fism" else None,
        "n_items":          n_items,
        "n_train":          int(X.shape[0]),
        "n_test":           len(rows_out),
        "fallback_count":   fallback_count,
    }
    if ndcgs:
        meta["r_precision"] = float(np.mean(r_precs))
        meta["ndcg"]        = float(np.mean(ndcgs))
        meta["clicks"]      = float(np.mean(clicks_list))
        print(f"\n{'='*40}")
        print(f"Submission : {output_name}")
        print(f"Playlists  : {len(ndcgs)}")
        print(f"R-Precision: {np.mean(r_precs):.6f}")
        print(f"NDCG       : {np.mean(ndcgs):.6f}")
        print(f"Clicks     : {np.mean(clicks_list):.6f}")
        print(f"{'='*40}\n")

    meta_path = output_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return output_name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iteración 3 - SLIM & FISM")
    parser.add_argument("--mode",     choices=["slim", "fism"], default="slim",
                        help="Algoritmo: SLIM (matriz S densa) o FISM (S=PQᵀ factorizada)")
    parser.add_argument("--epochs",   type=int,   default=500,  help="Épocas de entrenamiento")
    parser.add_argument("--lr",       type=float, default=0.01, help="Learning rate (Adam)")
    parser.add_argument("--lambda-a", type=float, default=5.0,  dest="lambda_a",
                        help="Coeficiente de regularización L2")
    parser.add_argument("--lambda-b", type=float, default=1.0,  dest="lambda_b",
                        help="Coeficiente de regularización L1")
    parser.add_argument("--factors",  type=int,   default=64,
                        help="Dimensión del espacio latente para FISM")
    args = parser.parse_args()
    generate_slim(
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        lambda_a=args.lambda_a,
        lambda_b=args.lambda_b,
        factors=args.factors,
    )
