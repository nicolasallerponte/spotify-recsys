"""
Iteración 1 — Neighbourhood-based Collaborative Filtering
==========================================================
Implementa dos variantes:

  User-based (Eq. 1):
      r̂(u,i) = Σ_{v ∈ Vᵤ} sim(u,v) · r(v,i)

  Item-based (Eq. 2):
      r̂(u,i) = Σ_{j ∈ Jᵢ} sim(i,j) · r(u,j)

En ambos casos la similitud es coseno y k es un hiperparámetro.

Uso:
    python src/knn.py --mode user --k 50
    python src/knn.py --mode item --k 50
"""

import argparse
import csv
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Set

import numpy as np
from scipy.sparse import load_npz, csr_matrix, csc_matrix, diags, issparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEAM_NAME = "Jacobo_Cousillas_Xaime_Paz_Nicolas_Aller"
TEAM_EMAIL = "jacobo.cousillas@udc.es_xaime.paz.ollero@udc.es_nicolas.aller@udc.es"
RECOMMENDATIONS_COUNT = 500


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def to_dense_1d(x) -> np.ndarray:
    if issparse(x):
        return np.asarray(x.todense()).flatten().astype(np.float32)
    return np.asarray(x).flatten().astype(np.float32)


def normalize_rows(matrix: csr_matrix) -> csr_matrix:
    norms = np.sqrt(np.array(matrix.power(2).sum(axis=1)).flatten())
    norms[norms == 0] = 1.0
    return diags(1.0 / norms) @ matrix


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


# ---------------------------------------------------------------------------
# User-based KNN
# ---------------------------------------------------------------------------

def recommend_user_based(
    seed_indices: list,
    matrix_norm_csr: csr_matrix,
    matrix_norm_csc: csc_matrix,
    idx_to_track: dict,
    seed_uris: Set[str],
    track_to_idx: dict,
    k: int,
) -> list:
    if not seed_indices:
        return []

    n_seed = len(seed_indices)
    seed_norm_val = 1.0 / np.sqrt(n_seed)

    # matrix_norm_csr: (n_playlists × n_tracks)
    # matrix_norm_csc[:, seed_indices] → (n_playlists × n_seed): playlists que tienen esos tracks
    seed_cols = matrix_norm_csc[:, seed_indices]
    sims_dense = to_dense_1d(seed_cols.sum(axis=1)) * seed_norm_val  # (n_playlists,)

    nonzero_indices = np.where(sims_dense > 0)[0]
    if len(nonzero_indices) == 0:
        return []

    nonzero_sims = sims_dense[nonzero_indices]

    if k >= len(nonzero_indices):
        topk_local = np.argsort(-nonzero_sims)
    else:
        topk_local = np.argpartition(-nonzero_sims, k)[:k]
        topk_local = topk_local[np.argsort(-nonzero_sims[topk_local])]

    topk_indices = nonzero_indices[topk_local]
    topk_sims = nonzero_sims[topk_local].astype(np.float32)

    k_actual = len(topk_sims)
    weights_row = csr_matrix(
        (topk_sims, ([0] * k_actual, np.arange(k_actual))),
        shape=(1, k_actual),
        dtype=np.float32,
    )
    neighbor_matrix = matrix_norm_csr[topk_indices]   # (k × n_tracks)
    scores = to_dense_1d(weights_row @ neighbor_matrix)

    return top500_from_scores(scores, seed_indices, idx_to_track)


# ---------------------------------------------------------------------------
# Item-based KNN
# ---------------------------------------------------------------------------

def recommend_item_based(
    seed_indices: list,
    item_norm_csr: csr_matrix,
    item_norm_csc: csc_matrix,
    idx_to_track: dict,
    seed_uris: Set[str],
    track_to_idx: dict,
    k: int,
) -> list:
    """
    item_norm_csr: (n_tracks × n_playlists) con filas normalizadas.
    Cada track es un vector sobre el espacio de playlists.

    score(i) = item_norm[i] · centroide_semilla
             = Σ_{j ∈ semilla} sim(i, j)

    centroide_semilla = Σ_{j ∈ semilla} item_norm[j]  →  vector (n_playlists,)

    Truco CSC: solo multiplicamos contra las columnas (playlists) donde
    el centroide es no-nulo, que son las playlists que contienen al menos
    un track de la semilla.
    """
    if not seed_indices:
        return []

    # Limitar semilla a k tracks si k < |semilla|
    if k < len(seed_indices):
        seed_norms = np.array(
            item_norm_csr[seed_indices].power(2).sum(axis=1)
        ).flatten()
        topk_seed_local = np.argpartition(-seed_norms, k)[:k]
        effective_seed = [seed_indices[i] for i in topk_seed_local]
    else:
        effective_seed = seed_indices

    # item_norm_csr[effective_seed]: (|eff_seed| × n_playlists)
    seed_vecs = item_norm_csr[effective_seed]               # (|eff_seed| × n_playlists)
    seed_centroid = to_dense_1d(seed_vecs.sum(axis=0))      # (n_playlists,)

    # Solo playlists con centroide > 0
    nonzero_pl = np.where(seed_centroid > 0)[0]
    if len(nonzero_pl) == 0:
        return []

    # item_norm_csc tiene shape (n_tracks × n_playlists)
    # item_norm_csc[:, nonzero_pl] → (n_tracks × |nonzero_pl|)
    sub_item = item_norm_csc[:, nonzero_pl]                 # CSC slice eficiente
    sub_centroid = seed_centroid[nonzero_pl].astype(np.float32)

    # scores[i] = sub_item[i] · sub_centroid → (n_tracks,)
    scores = to_dense_1d(sub_item @ sub_centroid)

    return top500_from_scores(scores, seed_indices, idx_to_track)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_knn(mode: str = "user", k: int = 50):
    start_time = time.time()

    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    TEST_ZIP_PATH = BASE_DIR / "data" / "raw" / "spotify_test_playlists.zip"
    OUTPUT_CSV_PATH = BASE_DIR / "submissions" / f"iteracion_1_knn_{mode}_k{k}.csv"
    METADATA_PATH = BASE_DIR / "submissions" / f"iteracion_1_knn_{mode}_k{k}_info.json"

    logging.info("Cargando matriz y mapeos...")
    try:
        matrix = load_npz(PROCESSED_DIR / "user_item_matrix.npz")
        with open(PROCESSED_DIR / "track_to_idx.json", "r", encoding="utf-8") as f:
            track_to_idx = json.load(f)
        idx_to_track = {int(v): k_ for k_, v in track_to_idx.items()}
    except FileNotFoundError as e:
        logging.error(f"Archivo no encontrado. Ejecuta data_loader.py primero. {e}")
        return

    n_playlists, n_tracks = matrix.shape
    logging.info(f"Matriz cargada: {n_playlists} playlists × {n_tracks} tracks")

    logging.info("Normalizando y convirtiendo matriz...")
    matrix_float = matrix.astype(np.float32)

    # matrix_norm_csr: (n_playlists × n_tracks), filas normalizadas → user-based
    matrix_norm_csr = normalize_rows(matrix_float)
    matrix_norm_csc = matrix_norm_csr.tocsc()

    if mode == "item":
        logging.info("Preparando vista item-based...")
        # item_norm_csr: (n_tracks × n_playlists), filas normalizadas → item-based
        matrix_T = matrix_norm_csr.T.tocsr()   # (n_tracks × n_playlists)
        item_norm_csr = normalize_rows(matrix_T)
        item_norm_csc = item_norm_csr.tocsc()  # columnas = playlists (990k)
        logging.info(f"item_norm shape: {item_norm_csr.shape}")
    else:
        item_norm_csr = item_norm_csc = None

    logging.info("Listo.")

    with open(PROCESSED_DIR / "popular_tracks.json", "r") as f:
        popular_tracks_raw = json.load(f)
    popular_uris_fallback = [uri for uri, _ in popular_tracks_raw]

    logging.info(f"Modo: {mode.upper()}-based | k={k}")
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_playlists = 0
    fallback_count = 0

    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['team_info', TEAM_NAME, TEAM_EMAIL])
        writer.writerow([])

        with zipfile.ZipFile(TEST_ZIP_PATH, "r") as zipf:
            with zipf.open("test_input_playlists.json") as f:
                data = json.loads(f.read())
                playlists = data.get("playlists", [])
                total = len(playlists)

                for i, playlist in enumerate(playlists):
                    if i % 500 == 0:
                        elapsed = time.time() - start_time
                        logging.info(f"Procesando playlist {i}/{total} ({elapsed:.1f}s)")

                    total_playlists += 1
                    pid = playlist["pid"]
                    seeds: Set[str] = {t["track_uri"] for t in playlist.get("tracks", [])}
                    seed_indices = build_seed_indices(seeds, track_to_idx)

                    if mode == "user":
                        recommendations = recommend_user_based(
                            seed_indices, matrix_norm_csr, matrix_norm_csc,
                            idx_to_track, seeds, track_to_idx, k
                        )
                    else:
                        recommendations = recommend_item_based(
                            seed_indices, item_norm_csr, item_norm_csc,
                            idx_to_track, seeds, track_to_idx, k
                        )

                    if len(recommendations) < RECOMMENDATIONS_COUNT:
                        fallback_count += 1
                        recs_set = set(recommendations)
                        for uri in popular_uris_fallback:
                            if uri not in seeds and uri not in recs_set:
                                recommendations.append(uri)
                            if len(recommendations) == RECOMMENDATIONS_COUNT:
                                break

                    writer.writerow([pid] + recommendations[:RECOMMENDATIONS_COUNT])

    execution_time = time.time() - start_time

    metadata = {
        "equipo": TEAM_NAME,
        "metodo": f"KNN {mode}-based (Iteración 1)",
        "hiperparametros": {"mode": mode, "k": k},
        "estadisticas": {
            "playlists_procesadas": total_playlists,
            "playlists_con_fallback": fallback_count,
            "recomendaciones_por_playlist": RECOMMENDATIONS_COUNT,
            "tiempo_ejecucion_segundos": round(execution_time, 2)
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logging.info(f"Procesadas {total_playlists} playlists en {execution_time:.2f}s")
    logging.info(f"Playlists con fallback a popularidad: {fallback_count}")
    logging.info(f"Output → {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Collaborative Filtering - Iteración 1")
    parser.add_argument("--mode", choices=["user", "item"], default="user",
                        help="user-based (Eq.1) o item-based (Eq.2)")
    parser.add_argument("--k", type=int, default=50,
                        help="Tamaño del vecindario")
    args = parser.parse_args()

    generate_knn(mode=args.mode, k=args.k)