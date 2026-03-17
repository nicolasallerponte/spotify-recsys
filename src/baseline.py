import zipfile
import json
import csv
import logging
import time
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz
from typing import Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TEAM_NAME = "Jacobo_Cousillas_Xaime_Paz_Nicolas_Aller"
TEAM_EMAIL = "jacobo.cousillas@udc.es_xaime.paz.ollero@udc.es_nicolas.aller@udc.es"
RECOMMENDATIONS_COUNT = 500


def generate_baseline():
    start_time = time.time()

    BASE_DIR = Path(__file__).resolve().parent.parent
    PROCESSED_DIR = BASE_DIR / "data" / "processed"
    TEST_ZIP_PATH = BASE_DIR / "data" / "raw" / "spotify_test_playlists.zip"
    OUTPUT_CSV_PATH = BASE_DIR / "submissions" / "iteracion_0_baseline.csv"
    METADATA_PATH = BASE_DIR / "submissions" / "iteracion_0_info.json"

    logging.info("Cargando matriz de entrenamiento y mapeos...")
    try:
        matrix = load_npz(PROCESSED_DIR / "user_item_matrix.npz")

        with open(PROCESSED_DIR / "track_to_idx.json", "r", encoding="utf-8") as f:
            track_to_idx = json.load(f)

        idx_to_track = {v: k for k, v in track_to_idx.items()}

    except FileNotFoundError as e:
        logging.error(f"Archivo no encontrado en data/processed/. Ejecuta data_loader.py primero. {e}")
        return

    logging.info("Calculando ranking de popularidad...")
    popularity_counts = np.array(matrix.sum(axis=0)).flatten()
    sorted_indices = np.argsort(-popularity_counts)
    global_top_uris = [idx_to_track[idx] for idx in sorted_indices[:2000]]

    logging.info("Procesando test_input_playlists.json...")
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_playlists = 0

    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['team_info', TEAM_NAME, TEAM_EMAIL])
        writer.writerow([])

        with zipfile.ZipFile(TEST_ZIP_PATH, "r") as zipf:
            with zipf.open("test_input_playlists.json") as f:
                data = json.loads(f.read())

                for playlist in data.get("playlists", []):
                    total_playlists += 1
                    pid = playlist["pid"]
                    seeds: Set[str] = {t["track_uri"] for t in playlist.get("tracks", [])}

                    recommendations = []
                    for uri in global_top_uris:
                        if uri not in seeds:
                            recommendations.append(uri)
                        if len(recommendations) == RECOMMENDATIONS_COUNT:
                            break

                    writer.writerow([pid] + recommendations)

    execution_time = time.time() - start_time

    metadata = {
        "equipo": TEAM_NAME,
        "metodo": "Popularity Baseline (Iteración 0)",
        "estadisticas": {
            "playlists_procesadas": total_playlists,
            "recomendaciones_por_playlist": RECOMMENDATIONS_COUNT,
            "tiempo_ejecucion_segundos": round(execution_time, 2)
        }
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    logging.info(f"Procesadas {total_playlists} playlists en {execution_time:.2f}s → {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    generate_baseline()
