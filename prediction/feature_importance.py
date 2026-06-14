import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.ai_framework import compute_physical_attribution


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute physical sensitivity attribution for a clock dataset")
    parser.add_argument("input_csv", nargs="?", default="rb_clock_data.csv")
    parser.add_argument("--output", default="physical_attribution.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    attribution = compute_physical_attribution(df)
    out_df = pd.DataFrame(attribution)
    out_df.to_csv(args.output, index=False)
    print(out_df)


if __name__ == "__main__":
    main()
