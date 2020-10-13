import os
from argparse import Namespace
from deepac.explain.deepbind_scores import get_maxact
from deepac.explain.filter_contribs import get_filter_contribs
from deepac.explain.filter_ranking import get_filter_ranking
from deepac.explain.weblogos import get_weblogos
from deepac.explain.weblogos_extended import get_weblogos_ext
from deepac.explain.fasta2transfac import fa2transfac
from deepac.explain.IC_from_transfac import transfac2ic
from deepac.explain.motif_comparison import motif_compare


class ExplainTester:
    """
    ExplainTester class.

    """

    def __init__(self, n_cpus=8, additivity_check=False):
        self.n_cpus = n_cpus
        self.model = os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002_converted.h5")
        self.neg_fasta = os.path.join("deepac-tests", "sample-val-neg.fasta")
        self.pos_fasta = os.path.join("deepac-tests", "sample-val-pos.fasta")
        self.test_data = os.path.join("deepac-tests", "sample_val_data.npy")
        self.outpath = os.path.join("deepac-tests", "explain")
        self.additivity_check = additivity_check

    def test_maxact(self):
        """Test maxact (DeepBind) scores."""
        args = Namespace(model=self.model, test_data=self.test_data, nonpatho_test=self.neg_fasta,
                         patho_test=self.pos_fasta, out_dir=os.path.join(self.outpath, "maxact"),
                         n_cpus=self.n_cpus, do_lstm=False, inter_layer=1, chunk_size=500)
        get_maxact(args)
        assert (os.path.isfile(os.path.join(self.outpath, "maxact", "fasta",
                                            "deepbind_sample_val_data_motifs_filter_1.fasta"))), "Maxact failed."
        assert (os.path.isfile(os.path.join(self.outpath, "maxact", "filter_activations",
                                            "deepbind_sample_val_data_act_filter_1.csv"))), "Maxact failed."

    def test_fcontribs(self):
        """Test contribution (DeepLIFT) scores."""
        args = Namespace(model=self.model, w_norm=True, test_data=self.test_data, nonpatho_test=self.neg_fasta,
                         patho_test=self.pos_fasta, out_dir=os.path.join(self.outpath, "fcontribs"), ref_mode="N",
                         inter_neuron=None, chunk_size=500, all_occurrences=False,
                         do_lstm=False, inter_layer=1, easy_partial=True, partial=False, train_data=None,
                         no_check=(not self.additivity_check))
        get_filter_contribs(args)
        assert (os.path.isfile(os.path.join(self.outpath, "fcontribs", "fasta",
                                            "sample_val_data_motifs_filter_1.fasta"))), "Fcontribs failed."
        assert (os.path.isfile(os.path.join(self.outpath, "fcontribs", "filter_scores",
                                            "sample_val_data_rel_filter_1.csv"))), "Fcontribs failed."
        assert (os.path.isfile(os.path.join(self.outpath, "fcontribs", "nuc_scores",
                                            "sample_val_data_rel_filter_1_nucleotides.csv"))), "Fcontribs failed."

    def test_franking(self):
        """Test filter ranking."""
        scores_dir = os.path.join("deepac-tests", "explain", "fcontribs", "filter_scores")
        y = os.path.join("deepac-tests", "sample_val_labels.npy")
        y_pred = os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002-predictions-sample_val.npy")
        args = Namespace(mode="original", scores_dir=scores_dir, true_label=y, pred_label=y_pred,
                         out_dir=os.path.join(self.outpath, "franking"))
        get_filter_ranking(args)

        assert (os.path.isfile(os.path.join(self.outpath, "franking", "original",
                                            "ranking_filter_4_classes_nonpatho_filter_original.txt"))), \
            "Franking failed."
        assert (os.path.isfile(os.path.join(self.outpath, "franking", "original",
                                            "ranking_filter_4_classes_patho_filter_original.txt"))), \
            "Franking failed."
        assert (os.path.isfile(
            os.path.join(self.outpath, "franking", "original",
                         "boxplots_contribution_scores_filter_1_wo_zeros_4_classes_original.png"))), \
            "Franking failed."
        assert (os.path.isfile(os.path.join(self.outpath, "franking", "original",
                                            "distr_contribution_scores_filter_1_wo_zeros_4_classes_original.png"))), \
            "Franking failed."

    def test_fa2transfac(self):
        """Test fasta to transfac conversion."""
        in_dir = os.path.join(self.outpath, "fcontribs", "fasta")
        out_dir = os.path.join(self.outpath, "fcontribs", "transfac_w")
        w_dir = os.path.join(self.outpath, "fcontribs", "filter_scores")
        args = Namespace(in_dir=in_dir, out_dir=out_dir, weighting=True, weight_dir=w_dir)
        fa2transfac(args)
        assert (os.path.isfile(os.path.join(self.outpath, "fcontribs", "transfac_w",
                                            "sample_val_data_motifs_filter_1_seq_weighting.transfac"))),\
            "fa2transfac failed."

        out_dir = os.path.join(self.outpath, "fcontribs", "transfac")
        args = Namespace(in_dir=in_dir, out_dir=out_dir, weighting=False, weight_dir=w_dir)
        fa2transfac(args)
        assert (os.path.isfile(os.path.join(self.outpath, "fcontribs", "transfac",
                                            "sample_val_data_motifs_filter_1.transfac"))),\
            "fa2transfac failed."

    def test_weblogos(self):
        """Test sequence logos."""
        in_dir = os.path.join(self.outpath, "maxact", "fasta")
        out_dir = os.path.join(self.outpath, "maxact", "weblogos")
        args = Namespace(in_dir=in_dir, file_ext=".fasta", train_data=self.test_data, out_dir=out_dir)
        get_weblogos(args)
        assert (os.path.isfile(os.path.join(self.outpath, "maxact", "weblogos",
                                            "weblogo_deepbind_sample_val_data_motifs_filter_1.png"))), \
            "Weblogos failed."

        in_dir = os.path.join(self.outpath, "fcontribs", "transfac_w")
        out_dir = os.path.join(self.outpath, "fcontribs", "weblogos")
        args = Namespace(in_dir=in_dir, file_ext=".transfac", train_data=self.test_data, out_dir=out_dir)
        get_weblogos(args)
        assert (os.path.isfile(os.path.join(self.outpath, "fcontribs", "weblogos",
                                            "weblogo_sample_val_data_motifs_filter_1_seq_weighting.png"))),\
            "Weblogos failed."

    def test_weblogos_extended(self):
        """Test extended sequence logos."""
        fasta_dir = os.path.join(self.outpath, "fcontribs", "fasta")
        logo_dir = os.path.join(self.outpath, "fcontribs", "transfac_w")
        scores_dir = os.path.join(self.outpath, "fcontribs", "nuc_scores")
        out_dir = os.path.join(self.outpath, "fcontribs", "weblogos_ext")
        args = Namespace(fasta_dir=fasta_dir, scores_dir=scores_dir, logo_dir=logo_dir,
                         train_data=self.test_data, out_dir=out_dir, gain=250*512)
        get_weblogos_ext(args)
        assert (os.path.isfile(
            os.path.join(self.outpath, "fcontribs", "weblogos_ext",
                         "weblogo_extended_sample_val_data_motifs_filter_1_seq_weighting.png"))),\
            "Extended weblogos failed."

    def test_transfac2ic(self):
        """Test IC calculation."""
        in_file = os.path.join(self.outpath, "fcontribs", "transfac",
                               "sample_val_data_motifs_filter_1_acgt.transfac")
        out_file = os.path.join(self.outpath, "fcontribs", "sample_val_data_motifs_filter_1_ic.txt")
        args = Namespace(in_file=in_file,
                         train=self.test_data,
                         out_file=out_file)
        transfac2ic(args)
        assert (os.path.isfile(out_file)), "transfac2IC failed."

    def test_motif_compare(self):
        """Test motif comparison."""
        in_file1 = os.path.join(self.outpath, "fcontribs", "transfac",
                                "sample_val_data_motifs_filter_1_acgt.transfac")
        in_file2 = os.path.join(self.outpath, "fcontribs", "transfac_w",
                                "sample_val_data_motifs_filter_1_seq_weighting_acgt.transfac")
        out_dir = os.path.join(self.outpath, "fcontribs", "motif_compare")
        args = Namespace(in_file1=in_file1,
                         in_file2=in_file2,
                         train_data=self.test_data,
                         out_dir=out_dir,
                         extensively=False,
                         rc=False,
                         shift=False,
                         min_overlap=5)
        motif_compare(args)
        assert (os.path.isfile(os.path.join(out_dir, "correlation_motifs.txt"))), "Motif comparison failed."
