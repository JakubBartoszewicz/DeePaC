import numpy as np
import os
import scipy
import seaborn as sns
from deepac.predict import predict_fasta, predict_npy


def pred_fwd_rc(model, input_npy, output_fwd, output_rc):
    """Predict pathogenic potentials from a preprocessed numpy array and its reverse-complement."""
    predict_npy(model, input_npy, output_fwd, rc=False)
    predict_npy(model, input_npy, output_rc, rc=True)


def compare_rc(model, input_npy):
    sns.set_context("paper", font_scale=2.5)
    out = "{}_predictions.png".format(os.path.splitext(input_npy)[0])
    kind = "scatter"
    marg_opt = dict(bins=100)
    #joint_opt = dict(alpha=1.0)
    pred_fwd = "{}_predictions.npy".format(os.path.splitext(input_npy)[0])
    pred_rc = "{}_rc_predictions.npy".format(os.path.splitext(input_npy)[0])

    pred_fwd_rc(model, input_npy, output_fwd=pred_fwd, output_rc=pred_rc)

    fwd = np.load(pred_fwd)
    rc = np.load(pred_rc)

    print(scipy.stats.ks_2samp(fwd, rc))
    print(scipy.stats.spearmanr(fwd, rc))

    g = (sns.jointplot(x=fwd, y=rc, kind=kind, color='k', marginal_kws=marg_opt)) \
        .set_axis_labels("forward", "reverse-complement")

    g.ax_joint.set_xticks([0, 0.5, 1])
    g.ax_joint.set_yticks([0, 0.5, 1])

    g.savefig(out)
