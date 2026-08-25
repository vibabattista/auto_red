import json
import os
import random
import time
from datetime import datetime

from config import CONFIG
from utils import run_ga


def print_config_summary(cfg):
    print("=" * 60)
    print(f"run tag     : {cfg['run_tag']}")
    print(f"timestamp   : {cfg['timestamp']}")
    print("-" * 60)
    print("target")
    print(f"  model              : {cfg['target_model']}")
    print(f"  keyword            : {cfg['key_word']}")
    print(f"  defence prompt     : {cfg['system_prompt']}")
    print(f"  reasoning effort   : {cfg['target_reasoning_effort']}")
    print("mutator")
    print(f"  model              : {cfg['mutator_model']}")
    print(f"  reasoning effort   : {cfg['mutator_reasoning_effort']}")
    print(f"  system prompt      : {cfg['mutator_system']}")
    print("ga")
    print(f"  generations        : {cfg['generations']}")
    print(f"  pop_size           : {cfg['pop_size']}")
    print(f"  k                  : {cfg['k']}")
    print(f"  n_elite            : {cfg['n_elite']}")
    print(f"  n_parents          : {cfg['n_parents']}")
    print(f"  operator_weights   : {cfg['operator_weights']}")
    print(f"  rng_seed           : {cfg['rng_seed']}")
    print("evaluation")
    print(f"  seeds              : {len(cfg['seeds'])} provided")
    print("=" * 60)


def main():
    cfg = dict(CONFIG)
    cfg["timestamp"] = datetime.now().isoformat(timespec="seconds")
    cfg["run_tag"] = datetime.now().strftime("%Y%m%d-%H%M%S")

    if cfg["rng_seed"] is not None:
        random.seed(cfg["rng_seed"])

    if cfg["verbosity"] >= 1:
        print_config_summary(cfg)

    start = time.time()
    records, history = run_ga(
        cfg["seeds"],
        model=cfg["target_model"],
        key_word=cfg["key_word"],
        system_prompt=cfg["system_prompt"],
        reasoning_effort=cfg["target_reasoning_effort"],
        mutator_model=cfg["mutator_model"],
        mutator_system=cfg["mutator_system"],
        mutator_reasoning_effort=cfg["mutator_reasoning_effort"],
        operator_weights=cfg["operator_weights"],
        generations=cfg["generations"],
        pop_size=cfg["pop_size"],
        k=cfg["k"],
        n_elite=cfg["n_elite"],
        n_parents=cfg["n_parents"],
    )
    wall_time = time.time() - start

    best_record = max(records, key=lambda r: r["fitness"])

    if cfg["verbosity"] >= 1:
        print("-" * 60)
        print("best prompt found:")
        print(f"  fitness   : {best_record['fitness']:.2f}")
        print(f"  gen       : {best_record['gen']}")
        print(f"  op        : {best_record['op']}")
        print(f"  prompt    : {best_record['prompt']}")
        print(f"wall time   : {wall_time:.1f}s")
        print("=" * 60)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], f"run_{cfg['run_tag']}.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": cfg,
            "history": history,
            "records": records,
            "best": best_record,
            "wall_time": wall_time,
        }, f, indent=2)

    if cfg["verbosity"] >= 1:
        print(f"results written to {out_path}")

    return records, history


if __name__ == "__main__":
    main()
