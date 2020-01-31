def add_explain_parser(xparser):
    explain_subparsers = xparser.add_subparsers(help='DeePaC explain subcommands. See command --help for details.')

    parser_maxact = explain_subparsers.add_parser('maxact', help='Get DeepBind-like max-activation scores.')
    parser_maxact.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_maxact.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
    parser_maxact.add_argument("-N", "--nonpatho_test", required=True,
                               help="Nonpathogenic reads of the test data set (.fasta)")
    parser_maxact.add_argument("-P", "--patho_test", required=True, help="Pathogenic reads of"
                                                                         " the test data set (.fasta)")
    parser_maxact.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_maxact.add_argument("-n", "--n_cpus", dest="n_cpus", default=1, type=int, help="Number of CPU cores")
    parser_maxact.add_argument("-R", "--recurrent", dest="do_lstm", action="store_true",
                               help="Interpret elements of the LSTM output")
    parser_maxact.add_argument("-l", "--inter_layer", dest="inter_layer", default=1, type=int,
                               help="Perform calculations for this intermediate layer")
    parser_maxact.set_defaults(func=run_maxact)

    parser_fcontribs = explain_subparsers.add_parser('fcontribs', help='Get DeepLIFT/SHAP filter contribution scores.')
    parser_fcontribs.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_fcontribs.add_argument("-b", "--w_norm", action="store_true",
                                  help="Set flag if filter weight matrices should be mean-centered")
    parser_fcontribs.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
    parser_fcontribs.add_argument("-N", "--nonpatho_test", required=True,
                                  help="Nonpathogenic reads of the test data set (.fasta)")
    parser_fcontribs.add_argument("-P", "--patho_test", required=True, help="Pathogenic reads of the "
                                                                            "test data set (.fasta)")
    parser_fcontribs.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_fcontribs.add_argument("-r", "--ref_mode", default="N", choices=['N', 'GC', 'own_ref_file'],
                                  help="Modus to calculate reference sequences")
    parser_fcontribs.add_argument("-a", "--train_data",
                                  help="Train data (.npy), necessary to calculate reference sequences"
                                       " if ref_mode is 'GC'")
    parser_fcontribs.add_argument("-F", "--ref_seqs",
                                  help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
    parser_fcontribs.add_argument("-i", "--inter_neuron", nargs='*', dest="inter_neuron", type=int,
                                  help="Perform calculations for this intermediate neuron only")
    parser_fcontribs.add_argument("-l", "--inter_layer", dest="inter_layer", default=1, type=int,
                                  help="Perform calculations for this intermediate layer")
    parser_fcontribs.add_argument("-c", "--seq_chunk", dest="chunk_size", default=500, type=int,
                                  help="Sequence chunk size")
    parser_fcontribs.add_argument("-A", "--all-occurrences", dest="all_occurrences", action="store_true",
                                  help="Extract contributions for all occurrences of a filter "
                                       "per read (Default: max only)")
    parser_fcontribs.add_argument("-R", "--recurrent", dest="do_lstm", action="store_true",
                                  help="Interpret elements of the LSTM output")
    partial_group = parser_fcontribs.add_mutually_exclusive_group(required=False)
    partial_group.add_argument("-p", "--partial", dest="partial", action="store_true",
                               help="Calculate partial nucleotide contributions per filter")
    partial_group.add_argument("-e", "--easy_partial", dest="easy_partial", action="store_true",
                               help="Calculate easy partial nucleotide contributions per filter. "
                                    "Works for the first convolutional layer only.")
    parser_fcontribs.set_defaults(func=run_fcontribs)

    parser_franking = explain_subparsers.add_parser('franking', help='Generate filter rankings.')
    parser_franking.add_argument("-m", "--mode", default="original", choices=["original", "rel_true_class",
                                                                              "rel_pred_class"],
                                 help="Use original filter scores or normalize scores relative to "
                                      "true or predicted classes.")
    parser_franking.add_argument("-f", "--scores_dir", required=True,
                                 help="Directory containing filter contribution scores (.csv)")
    parser_franking.add_argument("-y", "--true_label", required=True, help="File with true read labels (.npy)")
    parser_franking.add_argument("-p", "--pred_label", required=True, help="File with predicted read labels (.npy)")
    parser_franking.add_argument("-o", "--out_dir", required=True, help="Output directory")
    parser_franking.set_defaults(func=run_franking)

    parser_fa2transfac = explain_subparsers.add_parser('fa2transfac', help='Calculate transfac from fasta files.')
    parser_fa2transfac.add_argument("-i", "--in_dir", required=True, help="Directory containing motifs per filter "
                                                                          "(.fasta)")
    parser_fa2transfac.add_argument("-o", "--out_dir", required=True, help="Output directory")
    parser_fa2transfac.add_argument("-w", "--weighting", default=False, action="store_true",
                                    help="Weight sequences by their DeepLIFT score")
    parser_fa2transfac.add_argument("-d", "--weight_dir",
                                    help="Directory containing the DeepLIFT scores per filter "
                                         "(only required if --weighting is chosen)")
    parser_fa2transfac.set_defaults(func=run_fa2transfac)

    parser_weblogos= explain_subparsers.add_parser('weblogos', help='Get sequence logos.')
    parser_weblogos.add_argument("-i", "--in_dir", required=True, help="Directory containing motifs per filter")
    parser_weblogos.add_argument("-f", "--file_ext", default=".transfac", choices=['.fasta', '.transfac'],
                                 help="Extension of file format of input files (.fasta or .transfac)")
    parser_weblogos.add_argument("-t", "--train_data",
                                 help="Training data set (.npy) to compute GC-content. N-padding lowers GC!")
    parser_weblogos.add_argument("-o", "--out_dir", required=True, help="Output directory")
    parser_weblogos.set_defaults(func=run_weblogos)

    parser_xlogos = explain_subparsers.add_parser('xlogos', help='Get extended sequence logos.')
    parser_xlogos.add_argument("-f", "--fasta_dir", required=True, help="Directory containing motifs "
                                                                        "per filter (.fasta)")
    parser_xlogos.add_argument("-s", "--scores_dir", required=True,
                               help="Directory containing nucleotide scores per filter (.csv)")
    parser_xlogos.add_argument("-l", "--logo_dir",
                               help="Directory containing motifs in weighted transfac format (only required if "
                                    "weighted weblogos should be created)")
    parser_xlogos.add_argument("-t", "--train_data", help="Training data set to compute GC-content")
    parser_xlogos.add_argument("-o", "--out_dir", required=True, help="Output directory")
    parser_xlogos.set_defaults(func=run_xlogos)

    parser_transfac2IC = explain_subparsers.add_parser('transfac2IC', help='Calculate information content '
                                                                           'from transfac files.')
    parser_transfac2IC.add_argument("-i", "--in_file", required=True, help="File containing all filter motifs "
                                                                           "in transfac format")
    parser_transfac2IC.add_argument("-t", "--train", required=True, help="Training data set (.npy) to normalize "
                                                                         "for GC-content")
    parser_transfac2IC.add_argument("-o", "--out_file", default=True, help="Name of the output file")
    parser_transfac2IC.set_defaults(func=run_transfac2IC)

    parser_mcompare = explain_subparsers.add_parser('mcompare', help='Compare motifs.')
    parser_mcompare.add_argument("-q", "--in_file1", required=True, help="File containing all filter motifs "
                                                                         "in transfac format")
    parser_mcompare.add_argument("-t", "--in_file2", required=True, help="File containing all filter motifs "
                                                                         "in transfac format")
    parser_mcompare.add_argument("-e", "--extensively", action="store_true", help="Compare every motif from "
                                                                                  "--in_file1 with "
                                                                                  "every motif from --in_file2; "
                                                                                  "default: compare only motifs "
                                                                                  "with the same ID")
    parser_mcompare.add_argument("-r", "--rc", action="store_true", help="Consider RC-complement of a motif")
    parser_mcompare.add_argument("-s", "--shift", action="store_true", help="Shift motifs to find best alignment")
    parser_mcompare.add_argument("-m", "--min_overlap", type=int, default=5, help="Minimal overlap between two motifs "
                                                                                  "if motifs are shifted to find the "
                                                                                  "best alignment (--shift)")
    parser_mcompare.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_mcompare.set_defaults(func=run_mcompare)

    return xparser


def run_maxact(args):
    print("maxact")


def run_fcontribs(args):
    if args.ref_mode == "GC" and args.train_data is None:
        raise ValueError(
            "Training data (--train_data) is required to build reference sequences with the same GC-content!")
    if args.ref_mode == "own_ref_file" and args.ref_seqs is None:
        raise ValueError("File with own reference sequences (--ref_seqs) is missing!")
    print("fcontribs")


def run_franking(args):
    print("franking")


def run_fa2transfac(args):
    if args.weighting and args.weight_dir is None:
        raise ValueError(
            "Sequence weighting is selected but the directory containg this data (--weight_dir) is missing!")
    print("fa2transfac")


def run_weblogos(args):
    print("weblogos")


def run_xlogos(args):
    print("xlogos")


def run_transfac2IC(args):
    print("transfac2IC")


def run_mcompare(args):
    print("mcompare")
