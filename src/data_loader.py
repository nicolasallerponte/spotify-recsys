import zipfile
import json
import logging
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.sparse import csr_matrix, save_npz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_dataset(zip_path: Path, processed_dir: Path):
    logging.info(f"Procesando dataset desde {zip_path}...")

    rows, cols = [], []
    track_counter = Counter()
    track_to_idx = {}
    playlist_to_idx = {}

    with zipfile.ZipFile(zip_path, "r") as zipf:
        json_files = [f for f in zipf.namelist() if f.endswith(".json")]

        for i, file_name in enumerate(json_files):
            if i % 100 == 0:
                logging.info(f"Leyendo slice {i}/{len(json_files)}...")

            with zipf.open(file_name) as f:
                data = json.loads(f.read())
                for playlist in data.get("playlists", []):
                    p_id = playlist["pid"]

                    if p_id not in playlist_to_idx:
                        playlist_to_idx[p_id] = len(playlist_to_idx)
                    p_idx = playlist_to_idx[p_id]

                    for track in playlist.get("tracks", []):
                        t_uri = track["track_uri"]
                        track_counter[t_uri] += 1

                        if t_uri not in track_to_idx:
                            track_to_idx[t_uri] = len(track_to_idx)

                        rows.append(p_idx)
                        cols.append(track_to_idx[t_uri])

    # 1. Guardar ranking de popularidad
    logging.info("Guardando ranking de popularidad...")
    popular_tracks = track_counter.most_common()
    with open(processed_dir / "popular_tracks.json", "w") as f:
        json.dump(popular_tracks, f)

    # 2. Crear y guardar matriz CSR
    logging.info("Creando matriz CSR...")
    rows_arr = np.array(rows, dtype=np.int32)
    cols_arr = np.array(cols, dtype=np.int32)
    values = np.ones(len(rows_arr), dtype=np.int8)

    matrix = csr_matrix(
        (values, (rows_arr, cols_arr)),
        shape=(len(playlist_to_idx), len(track_to_idx)),
        dtype='int8'
    )
    save_npz(processed_dir / "user_item_matrix.npz", matrix)

    # 3. Guardar diccionarios de mapeo
    logging.info("Guardando diccionarios de mapeo...")
    with open(processed_dir / "track_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(track_to_idx, f)

    with open(processed_dir / "playlist_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(playlist_to_idx, f)

    # 4. Metadatos
    n_playlists, n_tracks = matrix.shape
    n_nonzero = matrix.nnz
    density = (n_nonzero / (n_playlists * n_tracks)) * 100

    matrix_info = {
        "nombre_equipo": "Equipo_Cousillas_Paz_Aller",
        "dimensiones": {
            "playlists": n_playlists,
            "canciones_unicas": n_tracks
        },
        "elementos_no_nulos": n_nonzero,
        "densidad_porcentaje": f"{density:.6f}%",
        "promedio_canciones_por_playlist": f"{n_nonzero / n_playlists:.2f}",
        "archivo_origen": zip_path.name
    }

    with open(processed_dir / "matrix_info.json", "w", encoding="utf-8") as f:
        json.dump(matrix_info, f, indent=4, ensure_ascii=False)

    logging.info(f"Proceso finalizado. Densidad: {density:.6f}%")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    ZIP_TRAIN = BASE_DIR / "data" / "raw" / "spotify_train_dataset.zip"
    PROCESSED = BASE_DIR / "data" / "processed"
    PROCESSED.mkdir(parents=True, exist_ok=True)
    build_dataset(ZIP_TRAIN, PROCESSED)
