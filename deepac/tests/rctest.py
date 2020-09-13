import numpy as np
import os
import scipy
import seaborn as sns
from deepac.predict import predict_npy
from deepac.explain.rf_sizes import get_rf_size
import matplotlib.pyplot as plt


def pred_fwd_rc(model, input_npy, output_fwd, output_rc):
    """Predict pathogenic potentials from a preprocessed numpy array and its reverse-complement."""
    predict_npy(model, input_npy, output_fwd, rc=False)
    predict_npy(model, input_npy, output_rc, rc=True)


def compare_rc(model, input_npy, kind="scatter", alpha=0.5):
    sns.set_context("paper", font_scale=2.5)
    out = "{}_predictions.png".format(os.path.splitext(input_npy)[0])
    if kind != "kde":
        marg_opt = dict(bins=100, binrange=(0.0, 1.0))
    else:
        marg_opt = {}

    if kind != "reg":
        joint_opt = dict(alpha=alpha)
    else:
        joint_opt = {}
    pred_fwd = "{}_predictions.npy".format(os.path.splitext(input_npy)[0])
    pred_rc = "{}_rc_predictions.npy".format(os.path.splitext(input_npy)[0])

    conv_layer_ids = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)]
    if len(conv_layer_ids) > 0:
        conv_layer_idx = conv_layer_ids[-1]
        motif_length = get_rf_size(model, conv_layer_idx)
        print("Receptive field size: {}".format(motif_length))

    pred_fwd_rc(model, input_npy, output_fwd=pred_fwd, output_rc=pred_rc)

    fwd = np.load(pred_fwd)
    rc = np.load(pred_rc)

    print(scipy.stats.ks_2samp(fwd, rc))
    print(scipy.stats.spearmanr(fwd, rc))
    diff = np.abs(fwd - rc)
    print("Max difference: {}".format(np.max(diff)))
    print("Mean difference: {}".format(np.mean(diff)))
    print("Median difference: {}".format(np.median(diff)))

    g = (sns.jointplot(x=fwd, y=rc, kind=kind, color='k', joint_kws=joint_opt, marginal_kws=marg_opt)) \
        .set_axis_labels("forward", "reverse-complement")

    g.ax_joint.set_xticks([0, 0.5, 1])
    g.ax_joint.set_yticks([0, 0.5, 1])

    g.savefig(out)
    plt.clf()
