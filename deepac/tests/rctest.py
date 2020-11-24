import numpy as np
import os
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from deepac.predict import predict_npy
from deepac.explain.rf_sizes import get_rf_size
from termcolor import colored


def pred_fwd_rc(model, input_npy, output_fwd, output_rc, replicates=1):
    """Predict pathogenic potentials from a preprocessed numpy array and its reverse-complement."""
    y_fwd, _ = predict_npy(model, input_npy, output_fwd, rc=False, replicates=replicates)
    y_rc, _ = predict_npy(model, input_npy, output_rc, rc=True, replicates=replicates)
    return y_fwd, y_rc


def get_tolerance_from_mode(mode):
    tol_dict = {"numpy": (1e-05, 1e-08),
                # as in numpy.allclose
                "assert": (1e-07, 0.0),
                # as in numpy.testing.assert_allclose
                "pep": (1e-09, 0.0),
                # as in math.isclose
                "float16": (1e-05, 5e-04)
                # atol calculated as min. tolerance that allows for number up to 1 in IEE 754
                # https://stackoverflow.com/questions/872544/what-range-of-numbers-can-be-represented-in-a-16-32-and-64-bit-ieee-754-syste
                }

    return tol_dict[mode]


def test_all_close(fwd, rc, tol_mode, severe=True):
    warncolor = 'red' if severe else 'yellow'
    rtol, atol = get_tolerance_from_mode(tol_mode)
    if np.allclose(fwd, rc, rtol=rtol, atol=atol):
        print(colored("All close: HOLDS at rel. tol={r}, abs. tol={a}.".format(r=rtol, a=atol), 'green'))
        return True
    else:
        print(colored("All close: VIOLATED at rel. tol={r}, abs. tol={a}!".format(r=rtol, a=atol), warncolor))
        print(colored("If this model should be RC-symmetric, test again at higher precision, "
                      "e.g. --dtype-policy float64.", warncolor))
        return False


def compare_rc(model, input_npy, output, kind="scatter", alpha=0.5, replicates=1):
    sns.set_context("paper", font_scale=2.5)
    out = "{}.png".format(os.path.splitext(output)[0])
    if kind != "kde":
        marg_opt = dict(bins=100, binrange=(0.0, 1.0))
    else:
        marg_opt = {}

    if kind != "reg":
        joint_opt = dict(alpha=alpha)
    else:
        joint_opt = {}
    pred_fwd = output
    pred_rc = "{}_rc.npy".format(os.path.splitext(output)[0])

    conv_layer_ids = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)]
    if len(conv_layer_ids) > 0:
        conv_layer_idx = conv_layer_ids[-1]
        motif_length = get_rf_size(model, conv_layer_idx, verbose=True)
        print("Receptive field size: {}".format(motif_length))

    fwd, rc = pred_fwd_rc(model, input_npy, output_fwd=pred_fwd, output_rc=pred_rc, replicates=replicates)

    print(scipy.stats.ks_2samp(fwd, rc))
    print(scipy.stats.spearmanr(fwd, rc))
    diff = np.abs(fwd - rc)
    print("Max difference: {}".format(np.max(diff)))
    print("Mean difference: {}".format(np.mean(diff)))
    print("Median difference: {}".format(np.median(diff)))
    print("Std. dev.: {}".format(np.std(diff)))
    test_all_close(fwd, rc, "numpy", severe=False)
    test_all_close(fwd, rc, "float16", severe=True)

    g = (sns.jointplot(x=fwd, y=rc, kind=kind, color='k', joint_kws=joint_opt, marginal_kws=marg_opt)) \
        .set_axis_labels("forward", "reverse-complement")

    g.ax_joint.set_xticks([0, 0.5, 1])
    g.ax_joint.set_yticks([0, 0.5, 1])

    g.savefig(out)
    plt.clf()
