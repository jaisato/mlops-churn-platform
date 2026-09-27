"""Genera un CSV con datos sinteticos para exploracion o pruebas.

Uso: python scripts/generate_data.py --rows 5000 --out data/churn.csv
     python scripts/generate_data.py --rows 5000 --drift-shift 1.0 --out data/churn-drift.csv

`--drift-shift` produce un "mundo desplazado" (features y relacion con la etiqueta)
para ensayar el ciclo drift -> reentrenamiento con `--data` / `CHURN_TRAIN_DATA`.
"""

import argparse
import sys
from pathlib import Path

# Resuelto respecto a este fichero: funciona desde cualquier directorio de trabajo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.data.generator import TARGET, DriftSpec, generate_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drift-shift", type=float, default=0.0)
    parser.add_argument("--out", default="data/churn.csv")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.rows, args.seed, drift=DriftSpec.from_shift(args.drift_shift))
    df.to_csv(out, index=False)
    print(
        f"{len(df)} filas escritas en {out} (churn rate={df[TARGET].mean():.2%}, "
        f"drift shift={args.drift_shift})"
    )


if __name__ == "__main__":
    main()
