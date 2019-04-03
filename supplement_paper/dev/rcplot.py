import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context("paper", font_scale=2.5)
pathroot = "nn-ca2none-e002-predictions-"
out = "ncnn-s005"
kind = "scatter"
marg_opt = dict(bins=100)
joint_opt = dict(alpha=0.05)

fwd = np.concatenate((np.load("{}test_1.npy".format(pathroot)), np.load("{}test_2.npy".format(pathroot))))
rc = np.concatenate((np.load("{}test_1-rc.npy".format(pathroot)), np.load("{}test_2-rc.npy".format(pathroot))))

#scipy.stats.ks_2samp(fwd, rc)
#scipy.stats.spearmanr(fwd, rc)

g = (sns.jointplot(x=fwd, y=rc, kind=kind, color='k', marginal_kws=marg_opt, joint_kws=joint_opt))\
    .set_axis_labels("forward", "reverse-complement")

g.ax_joint.set_xticks([0, 0.5, 1])
g.ax_joint.set_yticks([0, 0.5, 1])

g.savefig(out+".png")