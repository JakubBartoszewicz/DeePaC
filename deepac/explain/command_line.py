"""@package deepac.explain.command_line
A DeePaC explain CLI. Support filter visualization and analysis tools.

"""
from deepac.explain.deepbind_scores import get_maxact
from deepac.explain.filter_contribs import get_filter_contribs
from deepac.explain.filter_ranking import get_filter_ranking
from deepac.explain.weblogos import get_weblogos
from deepac.explain.weblogos_extended import get_weblogos_ext
from deepac.explain.fasta2transfac import fa2transfac
from deepac.explain.IC_from_transfac import transfac2ic
from deepac.explain.motif_comparison import motif_compare


def add_explain_parser(xparser):
    """Parse DeePaC explain CLI arguments."""
    explain_subparsers = xparser.add_subparsers(help='DeePaC explain subcommands. See command --help for details.')

    parser_maxact = explain_subparsers.add_parser('maxact', help='Get DeepBind-like max-activation scores.')
    parser_maxact.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_maxact.add_argument("-t", "--test-data", required=True, help="Test data (.npy)")
    parser_maxact.add_argument("-N", "--nonpatho-test", required=True,
                               help="Nonpathogenic reads of the test data set (.fasta)")
    parser_maxact.add_argument("-P", "--patho-test", required=True, help="Pathogenic reads of"
                                                                         " the test data set (.fasta)")
    parser_maxact.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_maxact.add_argument("-n", "--n-cpus", dest="n_cpus", type=int, help="Number of CPU cores. Default: all.")
    parser_maxact.add_argument("-R", "--recurrent", dest="do_lstm", action="store_true",
                               help="Interpret elements of the LSTM output")
    parser_maxact.add_argument("-l", "--inter-layer", dest="inter_layer", default=1, type=int,
                               help="Perform calculations for this intermediate layer")
    parser_maxact.add_argument("-c", "--seq-chunk", dest="chunk_size", default=500, type=int,
                                  help="Sequence chunk size. Decrease for lower memory usage.")
    parser_maxact.set_defaults(func=run_maxact)

    parser_fcontribs = explain_subparsers.add_parser('fcontribs', help='Get DeepLIFT/SHAP filter contribution scores.')
    parser_fcontribs.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_fcontribs.add_argument("-b", "--w-norm", dest="w_norm", action="store_true",
                                  help="Set flag if filter weight matrices should be mean-centered")
    parser_fcontribs.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
    parser_fcontribs.add_argument("-N", "--nonpatho-test", required=True,
                                  help="Nonpathogenic reads of the test data set (.fasta)")
    parser_fcontribs.add_argument("-P", "--patho-test", required=True, help="Pathogenic reads of the "
                                                                            "test data set (.fasta)")
    parser_fcontribs.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_fcontribs.add_argument("-r", "--ref-mode", default="N", choices=['N', 'GC', 'own_ref_file'],
                                  help="Modus to calculate reference sequences")
    parser_fcontribs.add_argument("-a", "--train-data",
                                  help="Train data (.npy), necessary to calculate reference sequences"
                                       " if ref_mode is 'GC'")
    parser_fcontribs.add_argument("-F", "--ref-seqs",
                                  help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
    parser_fcontribs.add_argument("-i", "--inter-neuron", nargs='*', dest="inter_neuron", type=int,
                                  help="Perform calculations for this intermediate neuron only")
    parser_fcontribs.add_argument("-l", "--inter-layer", dest="inter_layer", default=1, type=int,
                                  help="Perform calculations for this intermediate layer")
    parser_fcontribs.add_argument("-c", "--seq-chunk", dest="chunk_size", default=500, type=int,
                                  help="Sequence chunk size. Decrease for lower memory usage.")
    parser_fcontribs.add_argument("-A", "--all-occurrences", dest="all_occurrences", action="store_true",
                                  help="Extract contributions for all occurrences of a filter "
                                       "per read (Default: max only)")
    parser_fcontribs.add_argument("-R", "--recurrent", dest="do_lstm", action="store_true",
                                  help="Interpret elements of the LSTM output")
    parser_fcontribs.add_argument("--no-check", dest="no_check", action="store_true",
                                  help="Disable additivity check.")
    partial_group = parser_fcontribs.add_mutually_exclusive_group(required=False)
    partial_group.add_argument("-p", "--partial", dest="partial", action="store_true",
                               help="Calculate partial nucleotide contributions per filter.")
    partial_group.add_argument("-e", "--easy-partial", dest="easy_partial", action="store_true",
                               help="Calculate easy partial nucleotide contributions per filter. "
                                    "Works for the first convolutional layer only; disables all-occurences mode.")
    parser_fcontribs.set_defaults(func=run_fcontribs)

    parser_franking = explain_subparsers.add_parser('franking', help='Generate filter rankings.')
    parser_franking.add_argument("-m", "--mode", default="original", choices=["original", "rel_true_class",
                                                                              "rel_pred_class"],
                                 help="Use original filter scores or normalize scores relative to "
                                      "true or predicted classes.")
    parser_franking.add_argument("-f", "--scores-dir", required=True,
                                 help="Directory containing filter contribution scores (.csv)")
    parser_franking.add_argument("-y", "--true-label", required=True, help="File with true read labels (.npy)")
    parser_franking.add_argument("-p", "--pred-label", required=True, help="File with predicted read labels (.npy)")
    parser_franking.add_argument("-o", "--out-dir", required=True, help="Output directory")
    parser_franking.set_defaults(func=run_franking)

    parser_fa2transfac = explain_subparsers.add_parser('fa2transfac', help='Calculate transfac from fasta files.')
    parser_fa2transfac.add_argument("-i", "--in-dir", required=True, help="Directory containing motifs per filter "
                                                                          "(.fasta)")
    parser_fa2transfac.add_argument("-o", "--out-dir", required=True, help="Output directory")
    parser_fa2transfac.add_argument("-w", "--weighting", default=False, action="store_true",
                                    help="Weight sequences by their DeepLIFT score")
    parser_fa2transfac.add_argument("-W", "--weight-dir",
                                    help="Directory containing the DeepLIFT scores per filter "
                                         "(only required if --weighting is chosen)")
    parser_fa2transfac.set_defaults(func=run_fa2transfac)

    parser_weblogos = explain_subparsers.add_parser('weblogos', help='Get sequence logos.')
    parser_weblogos.add_argument("-i", "--in-dir", required=True, help="Directory containing motifs per filter")
    parser_weblogos.add_argument("-f", "--file-ext", default=".transfac", choices=['.fasta', '.transfac'],
                                 help="Extension of file format of input files (.fasta or .transfac)")
    parser_weblogos.add_argument("-t", "--train-data",
                                 help="Training data set (.npy) to compute GC-content. N-padding lowers GC!")
    parser_weblogos.add_argument("-o", "--out-dir", required=True, help="Output directory")
    parser_weblogos.set_defaults(func=run_weblogos)

    parser_xlogos = explain_subparsers.add_parser('xlogos', help='Get extended sequence logos.')
    parser_xlogos.add_argument("-i", "--fasta-dir", required=True, help="Directory containing motifs "
                                                                        "per filter (.fasta)")
    parser_xlogos.add_argument("-s", "--scores-dir", required=True,
                               help="Directory containing nucleotide scores per filter (.csv)")
    parser_xlogos.add_argument("-I", "--logo-dir",
                               help="Directory containing motifs in weighted transfac format (only required if "
                                    "weighted weblogos should be created)")
    parser_xlogos.add_argument("-G", "--gain", default=250 * 512, type=int,
                               help="Color saturation gain. Weblogo colors reach saturation when the average nt "
                                    "score=1/gain. Default: 128000. Recommended: input length * number of filters.")
    parser_xlogos.add_argument("-t", "--train-data", help="Training data set to compute GC-content")
    parser_xlogos.add_argument("-o", "--out-dir", required=True, help="Output directory")
    parser_xlogos.set_defaults(func=run_xlogos)

    parser_transfac2ic = explain_subparsers.add_parser('transfac2IC', help='Calculate information content '
                                                                           'from transfac files.')
    parser_transfac2ic.add_argument("-i", "--in-file", required=True, help="File containing all filter motifs "
                                                                           "in transfac format")
    parser_transfac2ic.add_argument("-t", "--train", required=True, help="Training data set (.npy) to normalize "
                                                                         "for GC-content")
    parser_transfac2ic.add_argument("-o", "--out-file", default=True, help="Name of the output file")
    parser_transfac2ic.set_defaults(func=run_transfac2ic)

    parser_mcompare = explain_subparsers.add_parser('mcompare', help='Compare motifs.')
    parser_mcompare.add_argument("-q", "--in-file1", required=True, help="File containing all filter motifs "
                                                                         "in transfac format")
    parser_mcompare.add_argument("-t", "--in-file2", required=True, help="File containing all filter motifs "
                                                                         "in transfac format")
    parser_mcompare.add_argument("-a", "--train-data",
                                 help="Training data (.npy), necessary to calculate background GC content")
    parser_mcompare.add_argument("-e", "--extensively", action="store_true", help="Compare every motif from "
                                                                                  "--in_file1 with "
                                                                                  "every motif from --in_file2; "
                                                                                  "default: compare only motifs "
                                                                                  "with the same ID")
    parser_mcompare.add_argument("-r", "--rc", action="store_true", help="Consider RC-complement of a motif")
    parser_mcompare.add_argument("-s", "--shift", action="store_true", help="Shift motifs to find best alignment")
    parser_mcompare.add_argument("-m", "--min-overlap", type=int, default=5, help="Minimal overlap between two motifs "
                                                                                  "if motifs are shifted to find the "
                                                                                  "best alignment (--shift)")
    parser_mcompare.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_mcompare.set_defaults(func=run_mcompare)

    return xparser


def run_maxact(args):
    """Get DeepBind-like max-activation scores."""
    get_maxact(args)


def run_fcontribs(args):
    """Get DeepLIFT/SHAP filter contribution scores."""
    if args.ref_mode == "GC" and args.train_data is None:
        raise ValueError(
            "Training data (--train_data) is required to build reference sequences with the same GC-content!")
    if args.ref_mode == "own_ref_file" and args.ref_seqs is None:
        raise ValueError("File with own reference sequences (--ref_seqs) is missing!")
    get_filter_contribs(args)


def run_franking(args):
    """Generate filter rankings."""
    get_filter_ranking(args)


def run_fa2transfac(args):
    """Calculate transfac from fasta files."""
    if args.weighting and args.weight_dir is None:
        raise ValueError(
            "Sequence weighting is selected but the directory containg this data (--weight_dir) is missing!")
    fa2transfac(args)


def run_weblogos(args):
    """Get sequence logos."""
    get_weblogos(args)


def run_xlogos(args):
    """Get extended sequence logos."""
    get_weblogos_ext(args)


def run_transfac2ic(args):
    """Calculate information content from transfac files."""
    transfac2ic(args)


def run_mcompare(args):
    """Compare motifs."""
    motif_compare(args)
