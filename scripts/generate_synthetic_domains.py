"""Generate the synthetic pseudo-domains for a dataset config.

    python scripts/generate_synthetic_domains.py --config configs/pacs.yaml
"""
import argparse
from dg2cd.data.synthetic_domains import generate_all_domains
from dg2cd.utils import load_config

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pacs.yaml")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate images that already exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"generator : {cfg.synthetic.generator}")
    print(f"source    : {cfg.dataset.source}")
    print(f"output    : {cfg.synthetic.cache_dir}\n")

    counts = generate_all_domains(cfg, overwrite=args.overwrite)

    for name, n in counts.items():
        kind = "train" if name in cfg.synthetic.train_domains else "valid"
        print(f"  {name:<8} [{kind}]  {n:>5} images written")
    print(f"\ntotal written: {sum(counts.values())}")

if __name__ == "__main__":
    main()