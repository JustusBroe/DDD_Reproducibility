
NOISES = [0.15, 0.0]
SEED = 0
EXTRA_SEEDS = [1, 2]                 
RES_PEAK = [6, 8, 10, 12, 14, 16, 20]
CNN_PEAK = [6, 8, 10, 12, 14, 16, 20]
RES_COARSE = [12, 10, 8, 14, 16, 6, 20, 4, 24, 32, 2, 40, 48, 56, 64]
CNN_COARSE = [10, 8, 12, 6, 14, 16, 20, 4, 24, 28, 32, 2, 40, 48, 56, 64, 1, 3, 5, 7, 9]
RES_ALL = list(range(1, 65))
CNN_ALL = list(range(1, 65))
def res_hours(k): return 1.564e6 * (2.4 + 2.5 * k / 64) * 1.15 / 3.6e6
def cnn_hours(k): return 0.5e6 * (1.2 + 0.8 * k / 64) / 3.6e6      # GUESS: CNN speed on B200 not measured yet


def cmd(arch, noise, aug, k, seed=0):
    if arch == "resnet":
        opt, size, ev, ck = "adam", "--epochs 4000", 100, 10
        steps_tag = "e4000"
    else:
        opt, size, ev, ck = "sgd", "--steps 500000", 60, 20
        steps_tag = "s500k"
    out = f"results/{arch}_{opt}_{steps_tag}_aug{aug}_p{noise}_k{k}_s{seed}.csv"
    return (f"python3 -u sweep.py --arch {arch} --opt {opt} {size} --n_eval {ev} --augs {aug} --noises {noise} "
            f"--ks {k} --seeds {seed} --noise_seed 0 --ckpt_dir ckpt --ckpt_every {ck} --out {out}"), \
           (res_hours(k) if arch == "resnet" else cnn_hours(k))


def build():
    tasks, seen = [], set()

    def add(arch, noise, aug, k, seed=0):
        c, h = cmd(arch, noise, aug, k, seed)
        if c not in seen:
            seen.add(c)
            tasks.append((c, h, arch))

    def tier(ks_res, ks_cnn, seeds):
        for seed in seeds:
            for noise in NOISES:
                for k in ks_res:
                    add("resnet", noise, 1, k, seed)
            for aug in (1, 0):
                for noise in NOISES:
                    for k in ks_cnn:
                        add("cnn", noise, aug, k, seed)

    tier(RES_COARSE, CNN_COARSE, [SEED])           
    n1 = len(tasks)
    tier(RES_PEAK, CNN_PEAK, EXTRA_SEEDS)           
    n2 = len(tasks)
    tier(RES_ALL, CNN_ALL, [SEED])                   
    build.marks = (n1, n2)
    return tasks


    tasks = build()
    with open("tasks.txt", "w") as f:
        f.write("\n".join(c for c, _, _ in tasks) + "\n")
    n1, n2 = build.marks
    print(f"{len(tasks)} runs written to tasks.txt")
    for name, a, b in (("1. coarse grid, seed 0", 0, n1), ("2. seeds %s near the peak" % EXTRA_SEEDS, n1, n2),
                       ("3. all other widths, seed 0", n2, len(tasks))):
        h = sum(t[1] for t in tasks[a:b])
        print(f"  tier {name}: {b-a} runs, ~{h:.0f} GPU-hours (~{h/6:.0f} h on 6 nodes)")
    print(f"everything: ~{sum(t[1] for t in tasks):.0f} GPU-hours")
