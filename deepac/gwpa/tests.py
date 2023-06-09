import os
from argparse import Namespace
from tensorflow.keras.models import load_model
import pandas as pd
from deepac.predict import predict_npy
from deepac.tests.datagen import generate_reads
from deepac.gwpa.fragment_genomes import frag_genomes
from deepac.gwpa.genome_pathogenicity import genome_map
from deepac.gwpa.gene_ranking import gene_rank
from deepac.gwpa.nt_contribs import nt_map
from deepac.gwpa.filter_activations import filter_activations
from deepac.gwpa.filter_enrichment import filter_enrichment
from deepac.gwpa.command_line import run_gff2genome
from tensorflow.keras.utils import get_custom_objects
from deepac.explain.rf_sizes import get_rf_size


class GWPATester:
    """
    GWPATester class.

    """

    def __init__(self, n_cpus=8, additivity_check=False, target_class=None):
        self.n_cpus = n_cpus
        self.model = os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002_converted.h5")
        self.outpath = os.path.join("deepac-tests", "gwpa")
        self.additivity_check = additivity_check
        self.target_class = target_class
        self.__gen_data()

    def __gen_data(self):
        """Generate test genome data."""
        if not os.path.exists(os.path.join(self.outpath, "genome_fasta")):
            os.makedirs(os.path.join(self.outpath, "genome_fasta"))
        if not os.path.exists(os.path.join(self.outpath, "genome")):
            os.makedirs(os.path.join(self.outpath, "genome"))
        if not os.path.exists(os.path.join(self.outpath, "genome_gff3")):
            os.makedirs(os.path.join(self.outpath, "genome_gff3"))

        self.gen_sample(1, 0.3, [9000, 3000, 7000, 1000])
        self.gen_sample(2, 0.7, [6000, 7000, 3000, 4000])

    def gen_sample(self, s_id, gc, contigs):
        """Generate sample genomes."""
        generate_reads(1, os.path.join(self.outpath, "genome_fasta", "sample_genome{i}.fasta".format(i=s_id)), gc=gc,
                       length=contigs[0], header="SAMPLE{i}1.1".format(i=s_id))
        generate_reads(1, os.path.join(self.outpath, "genome_fasta", "sample_genome{i}.fasta".format(i=s_id)), gc=gc,
                       length=contigs[1], header="SAMPLE{i}2.1".format(i=s_id), append=True)
        generate_reads(1, os.path.join(self.outpath, "genome_fasta", "sample_genome{i}.fasta".format(i=s_id)), gc=gc,
                       length=contigs[2], header="SAMPLE{i}3.1".format(i=s_id), append=True)
        generate_reads(1, os.path.join(self.outpath, "genome_fasta", "sample_genome{i}.fasta".format(i=s_id)),
                       gc=1.0 - gc, length=contigs[3], header="SAMPLE{i}4.1".format(i=s_id), append=True)

        df = pd.DataFrame([["SAMPLE{i}1.1".format(i=s_id), "Genbank", "region", "1", contigs[0], ".", "+", ".",
                            "ID=SAMPLE{i}1.1".format(i=s_id)],
                           ["SAMPLE{i}1.1".format(i=s_id), "Genbank", "gene", "51", contigs[0] - 50, ".", "+", ".",
                            "ID=gene-EXP1;gene=EXP1;product=EXP1;Note= This is a region coding for a gene"],
                           ["SAMPLE{i}2.1".format(i=s_id), "Genbank", "region", "1", contigs[1], ".", "+", ".",
                            "ID=SAMPLE{i}2.1".format(i=s_id)],
                           ["SAMPLE{i}3.1".format(i=s_id), "Genbank", "region", "1", contigs[2], ".", "+", ".",
                            "ID=SAMPLE{i}3.1".format(i=s_id)],
                           ["SAMPLE{i}4.1".format(i=s_id), "Genbank", "region", "1", contigs[3], ".", "+", ".",
                            "ID=SAMPLE{i}4.1".format(i=s_id)],
                           ["SAMPLE{i}4.1".format(i=s_id), "Genbank", "gene", "51", contigs[3] - 50, ".", "+", ".",
                            "ID=gene-OUT{i};gene=OUT{i};product=OUT{i};"
                            "Note= This is a region coding for a gene".format(i=s_id)]])
        df.to_csv(os.path.join(self.outpath, "genome_gff3", "sample_genome{i}.gff3".format(i=s_id)), sep="\t",
                  header=False, index=False)

    def test_fragment(self):
        """Test genome fragmentation."""
        args = Namespace(genomes_dir=os.path.join(self.outpath, "genome_fasta"),
                         read_len=250, shift=50, out_dir=os.path.join(self.outpath, "genome_frag"))
        frag_genomes(args)
        assert (os.path.isfile(os.path.join(self.outpath, "genome_frag", "sample_genome2_fragmented_genomes.fasta"))), \
            "Fragment genomes failed."
        assert (os.path.isfile(os.path.join(self.outpath, "genome_frag", "sample_genome2_fragmented_genomes.npy"))), \
            "Fragment genomes failed."

    def test_genomemap(self):
        """Test genome-wide phenotype potential map generation."""
        model = load_model(self.model, custom_objects=get_custom_objects())

        if not os.path.exists(os.path.join(self.outpath, "genome_frag_pred")):
            os.makedirs(os.path.join(self.outpath, "genome_frag_pred"))

        predict_npy(model, os.path.join(self.outpath, "genome_frag", "sample_genome1_fragmented_genomes.npy"),
                    os.path.join(self.outpath, "genome_frag_pred", "sample_genome1_fragmented_genomes_predictions.npy"))

        predict_npy(model, os.path.join(self.outpath, "genome_frag", "sample_genome2_fragmented_genomes.npy"),
                    os.path.join(self.outpath, "genome_frag_pred", "sample_genome2_fragmented_genomes_predictions.npy"))

        args = Namespace(dir_fragmented_genomes=os.path.join(self.outpath, "genome_frag"),
                         dir_fragmented_genomes_preds=os.path.join(self.outpath, "genome_frag_pred"),
                         genomes_dir=os.path.join(self.outpath, "genome"),
                         out_dir=os.path.join(self.outpath, "bedgraph"),
                         target_class=self.target_class)
        genome_map(args)
        assert (os.path.isfile(os.path.join(self.outpath, "bedgraph",
                                            "sample_genome2_fragmented_genomes_pathogenicity.bedgraph"))), \
            "Genome map failed."

    def test_granking(self):
        """Test gene ranking."""
        args = Namespace(patho_dir=os.path.join(self.outpath, "bedgraph"),
                         gff_dir=os.path.join(self.outpath, "genome_gff3"),
                         out_dir=os.path.join(self.outpath, "gene_rank"), extended=False, n_cpus=self.n_cpus)
        gene_rank(args)

    def test_ntcontribs(self):
        """Test nucleotide contribution map generation."""
        args = Namespace(model=self.model, dir_fragmented_genomes=os.path.join(self.outpath, "genome_frag"),
                         genomes_dir=os.path.join(self.outpath, "genome"),
                         out_dir=os.path.join(self.outpath, "bedgraph"), ref_mode="N", read_length=250,
                         chunk_size=500, gradient=False, no_check=(not self.additivity_check),
                         target_class=self.target_class)
        nt_map(args)
        assert (os.path.isfile(os.path.join(self.outpath, "bedgraph",
                                            "sample_genome2_fragmented_genomes_nt_contribs_map.bedgraph"))), \
            "Nt contribs failed."

        args = Namespace(model=self.model, dir_fragmented_genomes=os.path.join(self.outpath, "genome_frag"),
                         genomes_dir=os.path.join(self.outpath, "genome"),
                         train_data=os.path.join("deepac-tests", "sample_train_data.npy"),
                         out_dir=os.path.join(self.outpath, "bedgraph_gc"), ref_mode="GC", read_length=250,
                         chunk_size=100, gradient=False, no_check=(not self.additivity_check),
                         target_class=self.target_class)
        nt_map(args)
        assert (os.path.isfile(os.path.join(self.outpath, "bedgraph_gc",
                                            "sample_genome2_fragmented_genomes_nt_contribs_map.bedgraph"))), \
            "Nt contribs failed."

    def test_factiv(self):
        """Test filter activations."""
        args = Namespace(model=self.model,
                         test_data=os.path.join(self.outpath, "genome_frag", "sample_genome2_fragmented_genomes.npy"),
                         test_fasta=os.path.join(self.outpath, "genome_frag",
                                                 "sample_genome2_fragmented_genomes.fasta"),
                         out_dir=os.path.join(self.outpath, "factiv"), chunk_size=500, inter_layer=1, inter_neuron=[1])
        filter_activations(args)
        assert (os.path.isfile(os.path.join(self.outpath, "factiv",
                                            "sample_genome2_fragmented_genomes_filter_1.bed"))), \
            "Factiv failed."

        args = Namespace(model=self.model,
                         test_data=os.path.join(self.outpath, "genome_frag", "sample_genome2_fragmented_genomes.npy"),
                         test_fasta=os.path.join(self.outpath, "genome_frag",
                                                 "sample_genome2_fragmented_genomes.fasta"),
                         out_dir=os.path.join(self.outpath, "factiv0"), chunk_size=500, inter_layer=0, inter_neuron=[1])
        filter_activations(args)
        assert (os.path.isfile(os.path.join(self.outpath, "factiv0",
                                            "sample_genome2_fragmented_genomes_filter_1.bed"))), \
            "Factiv failed."

    def test_fenrichment(self):
        """Test filter enrichment analysis."""
        model = load_model(self.model, custom_objects=get_custom_objects())
        conv_layer_ids = [idx for idx, layer in enumerate(model.layers) if "Conv1D" in str(layer)]
        conv_layer_idx = conv_layer_ids[0]
        motif_length = model.get_layer(index=conv_layer_idx).get_weights()[0].shape[0]

        args = Namespace(bed_dir=os.path.join(self.outpath, "factiv"),
                         gff=os.path.join(self.outpath, "genome_gff3", "sample_genome2.gff3"),
                         out_dir=os.path.join(self.outpath, "fenrichment"),
                         motif_length=motif_length, n_cpus=self.n_cpus, extended=True, ttest=False,
                         min_overlap_factor=3)
        filter_enrichment(args)
        assert (len(os.listdir(os.path.join(self.outpath, "fenrichment"))) > 0), \
            "Fenrichment failed."

        # get just-before-pooling activs
        conv_layer_ids = [idx - 2 for idx, layer in enumerate(model.layers)
                          if "Global" in str(layer)]
        conv_layer_idx = conv_layer_ids[0]
        input_layer_id = [idx for idx, layer in enumerate(model.layers) if "Input" in str(layer)][0]
        motif_length = model.get_layer(index=input_layer_id).get_output_at(0).shape[1]
        args = Namespace(bed_dir=os.path.join(self.outpath, "factiv0"),
                         gff=os.path.join(self.outpath, "genome_gff3", "sample_genome2.gff3"),
                         out_dir=os.path.join(self.outpath, "fenrichment0"),
                         motif_length=motif_length, n_cpus=self.n_cpus, extended=True, ttest=True,
                         min_overlap_factor=3)
        filter_enrichment(args)
        assert (len(os.listdir(os.path.join(self.outpath, "fenrichment0"))) > 0), \
            "Fenrichment failed."

    def test_gff2genome(self):
        """Test .genome file creation."""
        args = Namespace(gff3_dir=os.path.join(self.outpath, "genome_gff3"),
                         out_dir=os.path.join(self.outpath, "genome"))
        run_gff2genome(args)
        assert (len(os.listdir(os.path.join(self.outpath, "genome"))) > 0), \
            "gff2genome failed."
