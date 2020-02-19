"""@package deepac.gwpa.command_line
A DeePaC gwpa CLI. Support GWPA tools.

"""
import warnings
from deepac.gwpa.fragment_genomes import frag_genomes
from deepac.gwpa.gene_ranking import gene_rank
from deepac.gwpa.genome_pathogenicity import genome_map
from deepac.gwpa.nt_contribs import nt_map
from deepac.gwpa.filter_activations import filter_activations
from deepac.gwpa.filter_enrichment import filter_enrichment


def add_gwpa_parser(gparser):
    gwpa_subparsers = gparser.add_subparsers(help='DeePaC gwpa subcommands. '
                                                  'See command --help for details.')
    parser_fragment = gwpa_subparsers.add_parser('fragment', help='Fragment genomes for analysis.')
    parser_fragment.add_argument("-g", "--genomes_dir", required=True, help="Directory containing genomes in .fasta")
    parser_fragment.add_argument("-r", "--read_len", default=250, type=int,
                                 help="Length of extracted reads/fragments (default: 250)")
    parser_fragment.add_argument("-s", "--shift", default=50, type=int,
                                 help="Shift to start with the next fragment (default:50)")
    parser_fragment.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_fragment.set_defaults(func=run_fragment)

    parser_genomemap = gwpa_subparsers.add_parser('genomemap', help='Generate a genome-wide phenotype potential map.')
    parser_genomemap.add_argument("-f", "--dir_fragmented_genomes", required=True,
                                  help="Directory containing the fragmented genomes (.fasta)")
    parser_genomemap.add_argument("-p", "--dir_fragmented_genomes_preds", required=True,
                                  help="Directory containing the predictions (.npy) of the fragmented genomes")
    parser_genomemap.add_argument("-g", "--genomes_dir", required=True, help="Directory containing genomes (.genome)")
    parser_genomemap.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_genomemap.set_defaults(func=run_genomemap)

    parser_granking = gwpa_subparsers.add_parser('granking', help='Generate gene rankings.')
    parser_granking.add_argument("-p", "--patho_dir", required=True,
                                 help="Directory containing the pathogenicity scores over all genomic "
                                      "regions per species (.bedgraph)")
    parser_granking.add_argument("-g", "--gff_dir", required=True,
                                 help="Directory containing the annotation data of the species (.gff)")
    parser_granking.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_granking.add_argument('-x', '--extended', dest='extended', action='store_true',
                                 help='Check for multiple CDSs per gene.')
    parser_granking.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores.", default=8, type=int)
    parser_granking.set_defaults(func=run_granking)

    parser_ntcontribs = gwpa_subparsers.add_parser('ntcontribs', help='Generate a genome-wide nt contribution map.')
    parser_ntcontribs.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_ntcontribs.add_argument("-f", "--dir_fragmented_genomes", required=True,
                                   help="Directory containing the fragmented genomes (.fasta)")
    parser_ntcontribs.add_argument("-g", "--genomes_dir", required=True, help="Directory containing genomes (.genome)")
    parser_ntcontribs.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_ntcontribs.add_argument("-r", "--ref_mode", default="N", choices=['N', 'GC', 'own_ref_file'],
                                   help="Modus to calculate reference sequences")
    parser_ntcontribs.add_argument("-a", "--train_data",
                                   help="Train data (.npy), necessary to calculate reference sequences "
                                        "if ref_mode is 'GC'")
    parser_ntcontribs.add_argument("-F", "--ref_seqs",
                                   help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
    parser_ntcontribs.add_argument("-L", "--read_length", dest="read_length", default=250, type=int,
                                   help="Fragment length")
    parser_ntcontribs.set_defaults(func=run_ntcontribs)

    parser_factiv = gwpa_subparsers.add_parser('factiv', help='Get filter activations.')
    parser_factiv.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_factiv.add_argument("-t", "--test_data", required=True, help="Test data (.npy)")
    parser_factiv.add_argument("-f", "--test_fasta", required=True, help="Reads of the test data set (.fasta)")
    parser_factiv.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_factiv.set_defaults(func=run_factiv)

    parser_fenrichment = gwpa_subparsers.add_parser('fenrichment', help='Run filter enrichment analysis.')
    parser_fenrichment.add_argument("-i", "--bed_dir", required=True,
                                    help="Input directory with filter activation values for a species (.bed)")
    parser_fenrichment.add_argument("-g", "--gff", required=True, help="Gff file of species")
    parser_fenrichment.add_argument("-o", "--out_dir", default=".", help="Output directory")
    parser_fenrichment.add_argument("-l", "--motif_length", default=15, type=int, help="Motif length")
    parser_fenrichment.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores.", default=8, type=int)
    parser_fenrichment.set_defaults(func=run_fenrichment)

    return gparser


def run_fragment(args):
    """Fragment genomes for analysis."""
    if args.shift > args.read_len:
        warnings.warn("Shift (" + str(args.shift) + ") is larger than read length (" + str(args.read_len) +
                      ")!")
    frag_genomes(args)


def run_genomemap(args):
    """Generate a genome-wide phenotype potential map."""
    genome_map(args)


def run_granking(args):
    """Generate gene rankings."""
    gene_rank(args)


def run_ntcontribs(args):
    """Generate a genome-wide nt contribution map."""
    nt_map(args)


def run_factiv(args):
    """Get filter activations."""
    filter_activations(args)


def run_fenrichment(args):
    """Run filter enrichment analysis."""
    filter_enrichment(args)
