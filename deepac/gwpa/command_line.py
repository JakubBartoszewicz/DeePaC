"""@package deepac.gwpa.command_line
A DeePaC gwpa CLI. Support GWPA tools.

"""
import warnings
import os
import re
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
    parser_fragment.add_argument("-g", "--genomes-dir", required=True, help="Directory containing genomes in .fasta")
    parser_fragment.add_argument("-r", "--read_len", default=250, type=int,
                                 help="Length of extracted reads/fragments (default: 250)")
    parser_fragment.add_argument("-s", "--shift", default=50, type=int,
                                 help="Shift to start with the next fragment (default:50)")
    parser_fragment.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_fragment.set_defaults(func=run_fragment)

    parser_genomemap = gwpa_subparsers.add_parser('genomemap', help='Generate a genome-wide phenotype potential map.')
    parser_genomemap.add_argument("-f", "--dir-fragmented-genomes", required=True,
                                  help="Directory containing the fragmented genomes (.fasta)")
    parser_genomemap.add_argument("-p", "--dir-fragmented-genomes-preds", required=True,
                                  help="Directory containing the predictions (.npy) of the fragmented genomes")
    parser_genomemap.add_argument("-g", "--genomes-dir", required=True, help="Directory containing genomes (.genome)")
    parser_genomemap.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_genomemap.set_defaults(func=run_genomemap)

    parser_granking = gwpa_subparsers.add_parser('granking', help='Generate gene rankings.')
    parser_granking.add_argument("-p", "--patho-dir", required=True,
                                 help="Directory containing the pathogenicity scores over all genomic "
                                      "regions per species (.bedgraph)")
    parser_granking.add_argument("-g", "--gff-dir", required=True,
                                 help="Directory containing the annotation data of the species (.gff)")
    parser_granking.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_granking.add_argument('-x', '--extended', dest='extended', action='store_true',
                                 help='Check for multiple CDSs per gene and unnamed genes.')
    parser_granking.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores.", type=int)
    parser_granking.set_defaults(func=run_granking)

    parser_ntcontribs = gwpa_subparsers.add_parser('ntcontribs', help='Generate a genome-wide nt contribution map.')
    parser_ntcontribs.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_ntcontribs.add_argument("-f", "--dir-fragmented-genomes", required=True,
                                   help="Directory containing the fragmented genomes (.fasta)")
    parser_ntcontribs.add_argument("-g", "--genomes-dir", required=True, help="Directory containing genomes (.genome)")
    parser_ntcontribs.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_ntcontribs.add_argument("-r", "--ref-mode", default="N", choices=['N', 'GC', 'own_ref_file'],
                                   help="Modus to calculate reference sequences")
    parser_ntcontribs.add_argument("-a", "--train-data",
                                   help="Train data (.npy), necessary to calculate reference sequences "
                                        "if ref_mode is 'GC'")
    parser_ntcontribs.add_argument("-F", "--ref-seqs",
                                   help="User provided reference sequences (.fasta) if ref_mode is 'own_ref_file'")
    parser_ntcontribs.add_argument("-L", "--read-length", dest="read_length", default=250, type=int,
                                   help="Fragment length")
    parser_ntcontribs.add_argument("-c", "--seq-chunk", dest="chunk_size", default=500, type=int,
                                   help="Sequence chunk size. Decrease for lower memory usage.")
    parser_ntcontribs.add_argument("-G", "--gradient", dest="gradient", action="store_true",
                                   help="Use Integrated Gradients instead of DeepLIFT.")
    parser_ntcontribs.add_argument("--no-check", dest="no_check", action="store_true",
                                   help="Disable additivity check.")
    parser_ntcontribs.set_defaults(func=run_ntcontribs)

    parser_factiv = gwpa_subparsers.add_parser('factiv', help='Get filter activations.')
    parser_factiv.add_argument("-m", "--model", required=True, help="Model file (.h5)")
    parser_factiv.add_argument("-t", "--test-data", required=True, help="Test data (.npy)")
    parser_factiv.add_argument("-f", "--test-fasta", required=True, help="Reads of the test data set (.fasta)")
    parser_factiv.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_factiv.add_argument("-l", "--inter-layer", dest="inter_layer", default=1, type=int,
                                  help="Perform calculations for this intermediate layer")
    parser_factiv.add_argument("-c", "--seq-chunk", dest="chunk_size", default=500, type=int,
                               help="Sequence chunk size. Decrease for lower memory usage.")
    parser_factiv.add_argument("-F", "--inter-neuron", nargs='*', dest="inter_neuron", type=int,
                               help="Perform calculations for this filter only")
    parser_factiv.set_defaults(func=run_factiv)

    parser_fenrichment = gwpa_subparsers.add_parser('fenrichment', help='Run filter enrichment analysis.')
    parser_fenrichment.add_argument("-i", "--bed-dir", required=True,
                                    help="Input directory with filter activation values for a species (.bed)")
    parser_fenrichment.add_argument("-g", "--gff", required=True, help="Gff file of species")
    parser_fenrichment.add_argument("-o", "--out-dir", default=".", help="Output directory")
    parser_fenrichment.add_argument("-l", "--motif-length", default=15, type=int, help="Motif length")
    parser_fenrichment.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores.", type=int)
    parser_fenrichment.add_argument('-x', '--extended', dest='extended', action='store_true',
                                    help='Check for multiple CDSs per gene and unnamed genes.')
    parser_fenrichment.set_defaults(func=run_fenrichment)

    parser_gff2genome = gwpa_subparsers.add_parser('gff2genome', help='Generate .genome files.')
    parser_gff2genome.add_argument('gff3_dir', help='Input directory.')
    parser_gff2genome.add_argument('out_dir', help='Output directory.')
    parser_gff2genome.set_defaults(func=run_gff2genome)

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


def run_gff2genome(args):
    """Generate .genome files."""
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for file in os.listdir(args.gff3_dir):
        if file.endswith(".gff") or file.endswith(".gff3"):
            pre, ext = os.path.splitext(file)
            out_file = pre + ".genome"
            gff2genome(os.path.join(args.gff3_dir, file), os.path.join(args.out_dir, out_file))


def gff2genome(gff3_path, out_path):
    """Generate a .genome file."""
    ptrn = re.compile(r'\sregion')
    out_lines = []
    with open(gff3_path) as in_file:
        for line in in_file:
            region = ptrn.search(line)
            if region:
                out_lines.append(line.split()[0] + "\t" + line.split()[4] + "\n")
    with open(out_path, 'w') as out_file:
        out_file.writelines(out_lines)

