import json
import zipfile
import logging
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def r_precision(prediction: List[str], ground_truth: Set[str]) -> float:
    if not ground_truth:
        return 0.0
    n = len(ground_truth)
    relevant_found = [t for t in prediction[:n] if t in ground_truth]
    return len(relevant_found) / n


def dcg(prediction: List[str], ground_truth: Set[str]) -> float:
    score = 0.0
    for i, track in enumerate(prediction):
        if track in ground_truth:
            score += 1.0 / np.log2(i + 2)
    return score


def ndcg(prediction: List[str], ground_truth: Set[str]) -> float:
    actual_dcg = dcg(prediction, ground_truth)
    ideal_prediction = ["relevant"] * len(ground_truth)
    ideal_dcg = dcg(ideal_prediction, set(ideal_prediction))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def song_clicks(prediction: List[str], ground_truth: Set[str]) -> float:
    for i, track in enumerate(prediction):
        if track in ground_truth:
            return i // 10
    return 51.0


def evaluate(submission_filename: str = None):
    BASE_DIR = Path(__file__).resolve().parent.parent
    TEST_ZIP_PATH = BASE_DIR / "data" / "raw" / "spotify_test_playlists.zip"

    # Si no se especifica fichero, usar el más reciente en submissions/
    if submission_filename is None:
        submissions = sorted((BASE_DIR / "submissions").glob("*.csv"))
        if not submissions:
            logging.error("No hay archivos CSV en submissions/.")
            return
        SUBMISSION_PATH = submissions[-1]
        logging.info(f"Evaluando el más reciente: {SUBMISSION_PATH.name}")
    else:
        SUBMISSION_PATH = BASE_DIR / "submissions" / submission_filename

    # 1. Extraer Ground Truth
    logging.info("Extrayendo ground truth desde test_eval_playlists.json...")
    ground_truths: Dict[str, Set[str]] = {}

    try:
        with zipfile.ZipFile(TEST_ZIP_PATH, "r") as zipf:
            with zipf.open("test_eval_playlists.json") as f:
                data = json.loads(f.read())
                for pl in data.get("playlists", []):
                    ground_truths[str(pl["pid"])] = {t["track_uri"] for t in pl.get("tracks", [])}
    except KeyError:
        logging.error("No se encontró test_eval_playlists.json dentro del ZIP.")
        return

    # 2. Evaluar submission
    logging.info(f"Calculando métricas para: {SUBMISSION_PATH.name}")
    metrics = {"r_prec": [], "ndcg": [], "clicks": []}

    try:
        with open(SUBMISSION_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0] == "team_info" or row[0] == "":
                    continue

                pid = row[0]
                preds = row[1:]

                if pid in ground_truths:
                    gt = ground_truths[pid]
                    metrics["r_prec"].append(r_precision(preds, gt))
                    metrics["ndcg"].append(ndcg(preds, gt))
                    metrics["clicks"].append(song_clicks(preds, gt))

    except FileNotFoundError:
        logging.error(f"No se encontró {SUBMISSION_PATH}.")
        return

    # 3. Resultados
    if metrics["ndcg"]:
        n = len(metrics["ndcg"])
        logging.info(f"RESULTADOS FINALES (sobre {n} playlists)")
        print(f"\n{'='*40}")
        print(f"Submission : {SUBMISSION_PATH.name}")
        print(f"Playlists  : {n}")
        print(f"R-Precision: {np.mean(metrics['r_prec']):.6f}")
        print(f"NDCG       : {np.mean(metrics['ndcg']):.6f}")
        print(f"Clicks     : {np.mean(metrics['clicks']):.6f}")
        print(f"{'='*40}\n")
    else:
        logging.warning("No se pudieron emparejar predicciones con ground truth.")


if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(filename)
