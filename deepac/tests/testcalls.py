import keras.backend as K
from keras.models import load_model
from deepac.predict import predict_fasta, predict_npy, filter_fasta
from deepac.nn_train import RCConfig, RCNet
from deepac.eval.eval import evaluate_reads
from deepac.convert import convert_cudnn
from deepac import preproc
from deepac.tests import datagen
from deepac import builtin_loading
from deepac import __file__
from deepac.explain.tests import ExplainTester
from deepac.gwpa.tests import GWPATester
import configparser
import os


def run_tests(n_cpus=8, n_gpus=0, explain=False, gwpa=False, all=False):
    """Generate sample data and run all tests."""
    if os.path.exists("deepac-tests"):
        print("Deleting previous test output...")
        for root, dirs, files in os.walk("deepac-tests", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    quick = explain or gwpa
    print("TEST: Generating data...")
    datagen.generate_sample_data()
    tester = Tester(n_cpus, n_gpus)
    print("TEST: Preprocessing data...")
    tester.test_preproc()
    print("TEST: Training...")
    tester.test_train(quick)
    if all or explain:
        print("TEST: Predicting...")
        tester.test_pred(quick)
    if all or not quick:
        print("TEST: Evaluating...")
        tester.test_eval()
        print("TEST: Converting...")
        tester.test_convert()
        print("TEST: Filtering...")
        tester.test_filter()
    if all or explain:
        explaintester = ExplainTester(n_cpus, n_gpus)
        print("X-TEST: Maxact (DeepBind)...")
        explaintester.test_maxact()
        print("X-TEST: Filter contributions (DeepLIFT)...")
        explaintester.test_fcontribs()
        print("X-TEST: Filter ranking...")
        explaintester.test_franking()
        print("X-TEST: fa2transfac...")
        explaintester.test_fa2transfac()
        print("X-TEST: Weblogos...")
        explaintester.test_weblogos()
        print("X-TEST: Extended weblogos...")
        explaintester.test_weblogos_extended()
        #print("X-TEST: transfac2IC...")
        #explaintester.test_transfac2ic()
        #print("X-TEST: Motif comparison...")
        #explaintester.test_motif_compare()

    if all or gwpa:
        gwpatester = GWPATester(n_cpus)
        print("X-TEST: Fragmenting genomes...")
        gwpatester.test_fragment()
        print("X-TEST: Genome-wide phenotype potential map...")
        gwpatester.test_genomemap()
        print("X-TEST: Gene ranking...")
        gwpatester.test_granking()
        print("X-TEST: Nucleotide contribution map...")
        gwpatester.test_ntcontribs()
        print("X-TEST: Filter activations...")
        gwpatester.test_factiv()
        print("X-TEST: Filter enrichment...")
        gwpatester.test_fenrichment()

    print("TEST: OK")


class Tester:

    """
    Tester class.

    """

    def __init__(self, n_cpus=8, n_gpus=0):
        self.n_cpus = n_cpus
        self.n_gpus = n_gpus

    def test_preproc(self):
        """Test preprocessing."""
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "preproc-train.ini"))
        preproc.preproc(config)
        assert (os.path.isfile(os.path.join("deepac-tests", "sample_train_data.npy"))), "Preprocessing failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "sample_train_labels.npy"))), "Preprocessing failed."

        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "preproc-val.ini"))
        preproc.preproc(config)
        assert (os.path.isfile(os.path.join("deepac-tests", "sample_val_data.npy"))), "Preprocessing failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "sample_val_labels.npy"))), "Preprocessing failed."

    def test_train(self, quick=False):
        """Test training."""
        print("TEST: Training (custom)...")
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "nn-test.ini"))
        paprconfig = RCConfig(config)
        self.__config_train(paprconfig).train()
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "nn-deepac-test-e001.h5"))), "Training failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "nn-deepac-test-e002.h5"))), "Training failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "training-deepac-test.csv"))), "Training failed."

        if not quick:
            print("TEST: Training (rapid)...")
            paprconfig = builtin_loading.get_rapid_training_config(self.n_cpus, self.n_gpus)
            self.__config_train(paprconfig).train()
            assert (os.path.isfile(os.path.join("deepac-tests", "img-rapid-cnn-logs",
                                                "nn-img-rapid-cnn-e001.h5"))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "img-rapid-cnn-logs",
                                                "nn-img-rapid-cnn-e002.h5"))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "img-rapid-cnn-logs",
                                                "training-img-rapid-cnn.csv"))), "Training failed."

            print("TEST: Training (sensitive)...")
            paprconfig = builtin_loading.get_sensitive_training_config(self.n_cpus, self.n_gpus)
            self.__config_train(paprconfig).train()
            assert (os.path.isfile(os.path.join("deepac-tests", "img-sensitive-lstm-logs",
                                                "nn-img-sensitive-lstm-e001.h5"))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "img-sensitive-lstm-logs",
                                                "nn-img-sensitive-lstm-e002.h5"))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "img-sensitive-lstm-logs",
                                                "training-img-sensitive-lstm.csv"))), "Training failed."



    def __config_train(self, paprconfig):
        """Set sample data paths and compile."""
        paprconfig.x_train_path = os.path.join("deepac-tests", "sample_train_data.npy")
        paprconfig.y_train_path = os.path.join("deepac-tests", "sample_train_labels.npy")
        paprconfig.x_val_path = os.path.join("deepac-tests", "sample_val_data.npy")
        paprconfig.y_val_path = os.path.join("deepac-tests", "sample_val_labels.npy")
        paprconfig.epoch_start = 0
        paprconfig.epoch_end = 2
        paprconfig.n_gpus = self.n_gpus
        paprconfig.n_cpus = self.n_cpus
        paprconfig.log_superpath = "deepac-tests"
        paprconfig.log_dir = paprconfig.log_superpath + "/{runname}-logs".format(runname=paprconfig.runname)

        if K.backend() == 'tensorflow':
            paprconfig.set_tf_session()
        paprnet = RCNet(paprconfig)
        paprnet.load_data()
        paprnet.compile_model()
        return paprnet

    def test_pred(self, quick=False):
        """Test predicting."""
        print("TEST: Predicting (custom)...")
        model = load_model(os.path.join("deepac-tests", "deepac-test-logs", "nn-deepac-test-e002.h5"))
        predict_npy(model, os.path.join("deepac-tests", "sample_val_data.npy"),
                    os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002-predictions-sample_val.npy"))
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test-e002-predictions-sample_val.npy"))), "Prediction failed."

        if not quick:
            print("TEST: Predicting (rapid)...")
            model = builtin_loading.load_rapid_model(self.n_cpus, self.n_gpus, log_path="deepac-tests")
            predict_npy(model, os.path.join("deepac-tests", "sample_val_data.npy"),
                        os.path.join("deepac-tests", "img-rapid-cnn-logs", "val-pred-rapid.npy"))
            assert (os.path.isfile(os.path.join("deepac-tests", "img-rapid-cnn-logs",
                                                "val-pred-rapid.npy"))), "Prediction failed."

            print("TEST: Predicting (sensitive)...")
            model = builtin_loading.load_sensitive_model(self.n_cpus, self.n_gpus, log_path="deepac-tests")
            predict_npy(model, os.path.join("deepac-tests", "sample_val_data.npy"),
                        os.path.join("deepac-tests", "img-sensitive-lstm-logs", "val-pred-sensitive.npy"))
            assert (os.path.isfile(os.path.join("deepac-tests", "img-sensitive-lstm-logs",
                                                "val-pred-sensitive.npy"))), "Prediction failed."


    def test_eval(self):
        """Test evaluating."""
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "eval-test.ini"))
        evaluate_reads(config)
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test-metrics.csv"))), "Evaluation failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test_2_sample_val_auc.png"))), "Evaluation failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test_2_sample_val_aupr.png"))), "Evaluation failed."

    def test_convert(self):
        """Test converting."""
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "nn-test.ini"))
        convert_cudnn(config, os.path.join("deepac-tests", "deepac-test-logs", "nn-deepac-test-e002.h5"), False)
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "nn-deepac-test-e002_converted.h5"))), "Conversion failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "nn-deepac-test-e002_weights.h5"))), "Conversion failed."

    def test_filter(self):
        """Test filtering."""
        model = load_model(os.path.join("deepac-tests", "deepac-test-logs", "nn-deepac-test-e002.h5"))
        predict_fasta(model, os.path.join("deepac-tests", "sample-val-pos.fasta"),
                      os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002-predictions-sample_val-pos.npy"))
        filter_fasta(os.path.join("deepac-tests", "sample-val-pos.fasta"),
                     os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002-predictions-sample_val-pos.npy"),
                     os.path.join("deepac-tests", "sample-val-pos-filtered.fasta"))
        assert (os.path.isfile(os.path.join("deepac-tests", "sample-val-pos-filtered.fasta"))), "Filtering failed."
