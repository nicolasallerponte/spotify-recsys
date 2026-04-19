"""
Iteración 2 — PureSVD (Matrix Factorization via Truncated SVD)

Implementa dos variantes:

  Inductive  — SVD solo sobre datos de entrenamiento; las playlists de test
               se proyectan al espacio latente.
  Transductive — SVD sobre train + test juntos; los vectores latentes de test
                 salen directamente de la descomposición.

Uso:
    python src/puresvd.py --mode inductive --k 100
    python src/puresvd.py --mode transductive --k 100
    python src/puresvd.py --mode inductive --k 100 --batch-size 500
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
from scipy.sparse import load_npz, csr_matrix, vstack
from scipy.sparse.linalg import svds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEAM_NAME = "Jacobo_Cousillas_Xaime_Paz_Nicolas_Aller"
TEAM_EMAIL = "jacobo.cousillas@udc.es_xaime.paz.ollero@udc.es_nicolas.aller@udc.es"
RECOMMENDATIONS_COUNT = 500


# Utilidades

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


def build_test_matrix(playlists: list, track_to_idx: dict, n_tracks: int) -> csr_matrix:
    rows, cols = [], []
    for i, playlist in enumerate(playlists):
        for t in playlist.get("tracks", []):
            uri = t["track_uri"]
            if uri in track_to_idx:
                rows.append(i)
                cols.append(track_to_idx[uri])
    n = len(playlists)
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(n, n_tracks), dtype=np.float32)


# Scoring individual (un playlist a la vez)

def recommend_inductive(seed_indices: list, Vt: np.ndarray) -> np.ndarray:
    """
    Proyecta la playlist de test al espacio latente y puntúa todos los tracks.

    La cancelación algebraica de Sigma simplifica:
        u_new = (r_new @ Vt.T) / s
        scores = (u_new * s) @ Vt  =  (r_new @ Vt.T) @ Vt

    Es decir: latent = Vt[:, seed_indices].sum(axis=1), scores = latent @ Vt
    """
    if not seed_indices:
        return np.zeros(Vt.shape[1], dtype=np.float32)
    latent = Vt[:, seed_indices].sum(axis=1)   # (k,)
    scores = latent @ Vt                        # (n_tracks,)
    return scores.astype(np.float32)


def recommend_transductive(user_latent: np.ndarray, s: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """
    La playlist ya tiene vector latente de la descomposición conjunta.
    scores = u * Sigma @ Vt
    """
    scores = (user_latent * s) @ Vt    # (n_tracks,)
    return scores.astype(np.float32)


# Scoring en batch (GEMM en vez de GEMV repetido)

def scores_batch_inductive(batch: list, track_to_idx: dict, n_tracks: int, Vt: np.ndarray) -> np.ndarray:
    """
    Calcula scores para un lote de playlists en una sola operación GEMM.

    Construye la matriz de semillas sparse R (batch_size × n_tracks) y calcula:
        latent_batch = R @ Vt.T        (sparse × dense → dense, batch_size × k)
        scores_batch = latent_batch @ Vt  (GEMM, batch_size × n_tracks)
    """
    rows, cols = [], []
    for local_i, playlist in enumerate(batch):
        for t in playlist.get("tracks", []):
            uri = t["track_uri"]
            if uri in track_to_idx:
                rows.append(local_i)
                cols.append(track_to_idx[uri])

    bs = len(batch)
    if rows:
        data = np.ones(len(rows), dtype=np.float32)
        R = csr_matrix((data, (rows, cols)), shape=(bs, n_tracks), dtype=np.float32)
        latent_batch = R @ Vt.T          # (bs, k)
    else:
        latent_batch = np.zeros((bs, Vt.shape[0]), dtype=np.float32)

    return (latent_batch @ Vt).astype(np.float32)   # (bs, n_tracks)


def scores_batch_transductive(U_batch: np.ndarray, s: np.ndarray, Vt: np.ndarray) -> np.ndarray:
    """
    Calcula scores para un lote de playlists transductive en una sola operación GEMM.
        scores_batch = (U_batch * s) @ Vt   (GEMM, batch_size × n_tracks)
    """
    return ((U_batch * s) @ Vt).astype(np.float32)  # (bs, n_tracks)


# Main

def generate_puresvd(mode: str = "inductive", k: int = 100, batch_size: int = 0):
    start_time = time.time()
    use_batch = batch_size > 0

    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    TEST_ZIP_PATH = BASE_DIR / "data" / "raw" / "spotify_test_playlists.zip"
    OUTPUT_CSV_PATH = BASE_DIR / "submissions" / f"iteracion_2_puresvd_{mode}_k{k}.csv"
    METADATA_PATH = BASE_DIR / "submissions" / f"iteracion_2_puresvd_{mode}_k{k}_info.json"

    logging.info("Cargando matriz y mapeos...")
    try:
        matrix = load_npz(PROCESSED_DIR / "user_item_matrix.npz")
        with open(PROCESSED_DIR / "track_to_idx.json", "r", encoding="utf-8") as f:
            track_to_idx = json.load(f)
        idx_to_track = {int(v): k_ for k_, v in track_to_idx.items()}
    except FileNotFoundError as e:
        logging.error(f"Archivo no encontrado. Ejecuta data_loader.py primero. {e}")
        return

    n_train, n_tracks = matrix.shape
    logging.info(f"Matriz cargada: {n_train} playlists × {n_tracks} tracks")

    with open(PROCESSED_DIR / "popular_tracks.json", "r") as f:
        popular_tracks_raw = json.load(f)
    popular_uris_fallback = [uri for uri, _ in popular_tracks_raw]

    logging.info("Cargando playlists de test...")
    with zipfile.ZipFile(TEST_ZIP_PATH, "r") as zipf:
        with zipf.open("test_input_playlists.json") as f:
            data = json.loads(f.read())
            playlists = data.get("playlists", [])

    logging.info(f"Playlists de test: {len(playlists)}")

    # Preparar matriz para SVD
    train = matrix.astype(np.float32)

    if mode == "transductive":
        logging.info("Construyendo matriz de test (transductive)...")
        test_sparse = build_test_matrix(playlists, track_to_idx, n_tracks)
        svd_matrix = vstack([train, test_sparse], format='csr')
        logging.info(f"Matriz combinada: {svd_matrix.shape}")
    else:
        svd_matrix = train

    # SVD truncada
    logging.info(f"Ejecutando SVD truncada k={k} (modo={mode})...")
    svd_start = time.time()
    U, s, Vt = svds(svd_matrix, k=k, which='LM')
    svd_time = time.time() - svd_start
    logging.info(f"SVD completada en {svd_time:.1f}s")

    # svds devuelve valores singulares en orden ascendente → invertir
    idx = np.argsort(-s)
    s = s[idx].astype(np.float32)
    Vt = Vt[idx].astype(np.float32)   # (k, n_tracks)

    if mode == "transductive":
        U = U[:, idx].astype(np.float32)
        U_test = U[n_train:].copy()    # (n_test, k)
        del U, svd_matrix, test_sparse, train
    else:
        del U, train

    batch_label = f" | batch={batch_size}" if use_batch else ""
    logging.info(f"Modo: {mode.upper()} | k={k}{batch_label}")
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_playlists = 0
    fallback_count = 0
    zero_seed_count = 0
    total = len(playlists)

    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['team_info', TEAM_NAME, TEAM_EMAIL])
        writer.writerow([])

        if use_batch:
            # --- Modo batch: GEMM por lotes ---
            for batch_start in range(0, total, batch_size):
                batch = playlists[batch_start:batch_start + batch_size]
                bs = len(batch)

                if batch_start % (batch_size * 4) == 0:
                    elapsed = time.time() - start_time
                    logging.info(f"Procesando batch {batch_start}/{total} ({elapsed:.1f}s)")

                if mode == "inductive":
                    scores_batch = scores_batch_inductive(batch, track_to_idx, n_tracks, Vt)
                else:
                    scores_batch = scores_batch_transductive(U_test[batch_start:batch_start + bs], s, Vt)

                for local_i, playlist in enumerate(batch):
                    total_playlists += 1
                    pid = playlist["pid"]
                    seeds: Set[str] = {t["track_uri"] for t in playlist.get("tracks", [])}
                    seed_indices = build_seed_indices(seeds, track_to_idx)

                    if not seed_indices:
                        zero_seed_count += 1

                    scores = scores_batch[local_i].copy()
                    recommendations = top500_from_scores(scores, seed_indices, idx_to_track)

                    if len(recommendations) < RECOMMENDATIONS_COUNT:
                        fallback_count += 1
                        recs_set = set(recommendations)
                        for uri in popular_uris_fallback:
                            if uri not in seeds and uri not in recs_set:
                                recommendations.append(uri)
                            if len(recommendations) == RECOMMENDATIONS_COUNT:
                                break

                    writer.writerow([pid] + recommendations[:RECOMMENDATIONS_COUNT])

        else:
            # --- Modo individual: GEMV playlist a playlist ---
            for i, playlist in enumerate(playlists):
                if i % 500 == 0:
                    elapsed = time.time() - start_time
                    logging.info(f"Procesando playlist {i}/{total} ({elapsed:.1f}s)")

                total_playlists += 1
                pid = playlist["pid"]
                seeds: Set[str] = {t["track_uri"] for t in playlist.get("tracks", [])}
                seed_indices = build_seed_indices(seeds, track_to_idx)

                if not seed_indices:
                    zero_seed_count += 1

                if mode == "inductive":
                    scores = recommend_inductive(seed_indices, Vt)
                else:
                    scores = recommend_transductive(U_test[i], s, Vt)

                recommendations = top500_from_scores(scores, seed_indices, idx_to_track)

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
        "metodo": f"PureSVD {mode} (Iteración 2)",
        "hiperparametros": {"mode": mode, "k": k, "batch_size": batch_size if use_batch else None},
        "estadisticas": {
            "playlists_procesadas": total_playlists,
            "playlists_con_fallback": fallback_count,
            "playlists_sin_semilla": zero_seed_count,
            "recomendaciones_por_playlist": RECOMMENDATIONS_COUNT,
            "tiempo_svd_segundos": round(svd_time, 2),
            "tiempo_total_segundos": round(execution_time, 2),
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logging.info(f"Procesadas {total_playlists} playlists en {execution_time:.2f}s")
    logging.info(f"Playlists con fallback a popularidad: {fallback_count}")
    logging.info(f"Playlists sin semilla: {zero_seed_count}")
    logging.info(f"Output → {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PureSVD - Iteración 2")
    parser.add_argument("--mode", choices=["inductive", "transductive"], default="inductive",
                        help="inductive (solo train) o transductive (train+test)")
    parser.add_argument("--k", type=int, default=100,
                        help="Número de factores latentes")
    parser.add_argument("--batch-size", type=int, default=0,
                        help="Tamaño de lote para scoring en batch (0 = desactivado). "
                             "Usa GEMM en vez de GEMV repetido → más rápido en CPU multicore. "
                             "Recomendado: 500. Aumenta uso de RAM en batch_size × n_tracks × 4 bytes.")
    args = parser.parse_args()

    generate_puresvd(mode=args.mode, k=args.k, batch_size=args.batch_size)
