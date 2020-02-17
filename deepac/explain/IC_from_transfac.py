import math
import numpy as np
from Bio.motifs import transfac
import os


def compute_mean_ic(pwm):
    """Compute mean information content of a PWM."""
    ic = 0.0
    for column in range(pwm.length):
        for base in pwm.alphabet:
            ic += pwm[base][column] * math.log2(pwm[base][column]/pwm.background[base])
    ic /= float(pwm.length)
    return ic


def transfac2ic(args):
    """Compute information content for each filter motif (.transfac)."""
    train_samples = np.load(args.train, mmap_mode='r')
    probs = np.mean(np.mean(train_samples, axis=1), axis = 0)
    # background
    bg = {'A': probs[0], 'C': probs[1], 'G': probs[2], 'T': probs[3]}

    # create output directory
    out_dir = os.path.dirname(args.out_file)
    if out_dir == "":
        out_dir = "."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load all filter motifs
    with open(args.in_file) as handle:
        records = transfac.read(handle)

    # for each motif compute IC:
    for m in records:

        pwm = m.counts.normalize(pseudocounts=bg)
        pwm.background = bg
    # pssm = pwm.log_odds(background = bg))
        ic = compute_mean_ic(pwm)
        with open(args.out_file, "a") as file:
            file.write(m.get("ID") + "\t" + str(ic) + "\n")

