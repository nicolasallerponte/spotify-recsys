"""
Iteración 1 — Neighbourhood-based Collaborative Filtering

Implementa dos variantes:

  User-based
  Item-based

En ambos casos la similitud es coseno y k es un hiperparámetro.

Uso:
    python src/knn.py --mode user --k 500
    python src/knn.py --mode item --k 500
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


# Utilidades

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


# User-based KNN

def recommend_user_based(
    seed_indices: list,
    matrix_norm_csr: csr_matrix,
    matrix_norm_csc: csc_matrix,
    idx_to_track: dict,
    k: int,
) -> list:
    if not seed_indices:
        return []

    n_seed = len(seed_indices)
    seed_norm_val = 1.0 / np.sqrt(n_seed)

    # Similitudes: extraer columnas semilla de CSC y sumar
    seed_cols = matrix_norm_csc[:, seed_indices]
    sims_dense = to_dense_1d(seed_cols.sum(axis=1)) * seed_norm_val

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
    neighbor_matrix = matrix_norm_csr[topk_indices]
    scores = to_dense_1d(weights_row @ neighbor_matrix)

    return top500_from_scores(scores, seed_indices, idx_to_track)


# Item-based KNN

def recommend_item_based(
    seed_indices: list,
    item_norm_csr: csr_matrix,
    item_norm_csc: csc_matrix,
    idx_to_track: dict,
    k: int,
) -> list:
    """
    Formulación correcta de item-based KNN:

        r̂(u,i) = Σ_{j ∈ Jᵢ} sim(i,j) · r(u,j)

    donde Jᵢ son los k vecinos más similares del item i,
    y r(u,j) = 1 si j está en la semilla.

    Equivalente invertido (eficiente):
    Para cada track semilla j, calculamos sus k vecinos más similares
    y acumulamos sim(vecino, j) en scores[vecino].

    Esto es correcto porque:
        score(i) = Σ_{j ∈ semilla} sim(i,j) · [i ∈ top-k vecinos de j]
    que es la misma cantidad vista desde j en vez de desde i.

    item_norm_csr: (n_tracks × n_playlists), filas normalizadas.
    Similitud coseno entre tracks i y j = item_norm_csr[i] · item_norm_csr[j].
    """
    if not seed_indices:
        return []

    n_tracks = item_norm_csr.shape[0]
    scores = np.zeros(n_tracks, dtype=np.float32)

    for j in seed_indices:
        # Vector del track semilla j: (1 × n_playlists)
        j_vec = item_norm_csr[j]  # sparse (1 × n_playlists)

        # Similitudes de j con todos los tracks: (n_tracks,)
        # = item_norm_csr @ j_vec.T
        # Usamos CSC para extraer solo playlists donde j_vec es no-nulo
        j_nonzero_pl = j_vec.indices  # playlists donde aparece j
        if len(j_nonzero_pl) == 0:
            continue

        sub_item = item_norm_csc[:, j_nonzero_pl]       # (n_tracks × |pl_j|)
        sub_j = np.asarray(j_vec[:, j_nonzero_pl].todense()).flatten().astype(np.float32)
        sims_j = to_dense_1d(sub_item @ sub_j)          # (n_tracks,) similitudes con j

        # Top-k vecinos de j (excluir j mismo)
        sims_j[j] = 0.0
        nonzero_mask = sims_j > 0
        nonzero_idx = np.where(nonzero_mask)[0]

        if len(nonzero_idx) == 0:
            continue

        nonzero_sims = sims_j[nonzero_idx]

        if k >= len(nonzero_idx):
            topk_local = np.argsort(-nonzero_sims)
        else:
            topk_local = np.argpartition(-nonzero_sims, k)[:k]
            topk_local = topk_local[np.argsort(-nonzero_sims[topk_local])]

        topk_item_indices = nonzero_idx[topk_local]
        topk_item_sims = nonzero_sims[topk_local]

        # Acumular similitudes en scores
        scores[topk_item_indices] += topk_item_sims

    return top500_from_scores(scores, seed_indices, idx_to_track)



# Main

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
    matrix_norm_csr = normalize_rows(matrix_float)
    matrix_norm_csc = matrix_norm_csr.tocsc()

    if mode == "item":
        logging.info("Preparando vista item-based...")
        matrix_T = matrix_norm_csr.T.tocsr()   # (n_tracks × n_playlists)
        item_norm_csr = normalize_rows(matrix_T)
        item_norm_csc = item_norm_csr.tocsc()
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
                            idx_to_track, k
                        )
                    else:
                        recommendations = recommend_item_based(
                            seed_indices, item_norm_csr, item_norm_csc,
                            idx_to_track, k
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
