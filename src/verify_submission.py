import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def verify(submission_filename: str = None):
    BASE_DIR = Path(__file__).resolve().parent.parent

    if submission_filename is None:
        submissions = sorted((BASE_DIR / "submissions").glob("*.csv"))
        if not submissions:
            logging.error("No hay archivos CSV en submissions/.")
            return
        SUBMISSION_FILE = submissions[-1]
        logging.info(f"Verificando el más reciente: {SUBMISSION_FILE.name}")
    else:
        SUBMISSION_FILE = BASE_DIR / "submissions" / submission_filename

    logging.info(f"Verificando formato de: {SUBMISSION_FILE.name}")

    try:
        with open(SUBMISSION_FILE, 'r') as f:
            lines = [l for l in f.readlines() if l.strip() and not l.startswith('team_info')]

        errors = 0
        pids_vistos = set()

        for i, line in enumerate(lines):
            parts = line.strip().split(',')
            pid = parts[0]
            tracks = parts[1:]

            if len(tracks) != 500:
                logging.error(f"PID {pid}: tiene {len(tracks)} tracks (deben ser 500)")
                errors += 1

            if len(set(tracks)) != len(tracks):
                logging.error(f"PID {pid}: contiene tracks duplicados")
                errors += 1

            if pid in pids_vistos:
                logging.error(f"PID {pid}: duplicado en el CSV")
                errors += 1
            pids_vistos.add(pid)

            if not all(t.strip().startswith('spotify:track:') for t in tracks):
                logging.error(f"PID {pid}: tracks con formato URI incorrecto")
                errors += 1

        if errors == 0:
            logging.info(f"OK — {len(pids_vistos)} playlists verificadas sin errores.")
        else:
            logging.warning(f"{errors} errores encontrados.")

    except FileNotFoundError:
        logging.error(f"No se encontró {SUBMISSION_FILE}.")


if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else None
    verify(filename)
