from pathlib import Path
import pickle

folder = Path.home() / "Downloads"

# Only load pickle files from a source you trust.
with (folder / "dd-cifar10-mcnn-p10-sgd-Mlist").open("rb") as f:
    results = pickle.load(f)

with (folder / "dd-cifar10-mcnn-p10-sgd-ks").open("rb") as f:
    ks = pickle.load(f)

print(results[0].keys())
print(results[0]["Test Error"].shape)

# First experiment, first model size, all recorded steps:
test_error = results[0]["Test Error"][0, :]