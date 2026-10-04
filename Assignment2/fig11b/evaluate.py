"""Teacher-forced evaluation of a fairseq translation checkpoint.

For each split it reports, over all non-padding target tokens (including EOS):
  loss  = mean negative log-likelihood in bits (what fairseq logs as nll_loss)
  error = 100 * fraction of tokens where argmax prediction != target
These match the columns of the authors' CSVs (test_loss, test_error, train_loss, train_error).
"""

import argparse
import json
import math
from pathlib import Path

import torch
from fairseq import checkpoint_utils, utils


@torch.no_grad()
def evaluate_split(model, task, split, max_tokens, use_cuda):
    task.load_dataset(split, combine=False, epoch=1)
    dataset = task.dataset(split)
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=max_tokens,
        max_positions=utils.resolve_max_positions(task.max_positions(), model.max_positions()),
        ignore_invalid_inputs=True,
        seed=0,
        num_workers=1,
    ).next_epoch_itr(shuffle=False)

    pad = task.target_dictionary.pad()
    nll_sum, n_correct, n_tokens = 0.0, 0, 0
    for sample in itr:
        if use_cuda:
            sample = utils.move_to_cuda(sample)
        net_output = model(**sample["net_input"])
        lprobs = model.get_normalized_probs(net_output, log_probs=True).float()
        target = sample["target"]
        mask = target.ne(pad)
        target_lprobs = lprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll_sum += -target_lprobs[mask].sum().item()
        n_correct += lprobs.argmax(-1).eq(target)[mask].sum().item()
        n_tokens += mask.sum().item()

    return {
        "loss": nll_sum / n_tokens / math.log(2),
        "error": 100.0 * (1.0 - n_correct / n_tokens),
        "tokens": n_tokens,
        "sentences": len(dataset),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--splits", default="test,train")
    p.add_argument("--expected-updates", type=int, default=80000)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--size", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.add_argument(
        "--check-only",
        action="store_true",
        help="exit 0 if the checkpoint has finished training, 1 otherwise",
    )
    args = p.parse_args()

    if args.check_only:
        if not args.checkpoint.exists():
            raise SystemExit(1)
        state = checkpoint_utils.load_checkpoint_to_cpu(str(args.checkpoint))
        raise SystemExit(0 if state["optimizer_history"][-1]["num_updates"] >= args.expected_updates else 1)
    if args.size is None or args.n is None or args.out is None:
        p.error("--size, --n and --out are required")

    state = checkpoint_utils.load_checkpoint_to_cpu(str(args.checkpoint))
    num_updates = state["optimizer_history"][-1]["num_updates"]
    if num_updates < args.expected_updates:
        raise SystemExit(
            f"{args.checkpoint} has {num_updates} updates < {args.expected_updates}; training incomplete"
        )

    models, _cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [str(args.checkpoint)], arg_overrides={"data": str(args.data)}, state=state
    )
    model = models[0]
    use_cuda = torch.cuda.is_available()
    model.eval()
    if use_cuda:
        model.cuda()

    result = {
        "n": args.n,
        "size": args.size,
        "seed": args.seed,
        "num_updates": num_updates,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    for split in args.splits.split(","):
        metrics = evaluate_split(model, task, split, args.max_tokens, use_cuda)
        print(f"{split}: {metrics}")
        for key, value in metrics.items():
            result[f"{split}_{key}"] = value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
