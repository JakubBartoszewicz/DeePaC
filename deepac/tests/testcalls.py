import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from deepac.predict import predict_fasta, predict_npy, filter_fasta
from deepac.nn_train import RCConfig, RCNet
from deepac.eval.eval import evaluate_reads
from deepac.convert import convert_cudnn
from deepac import preproc
from deepac.tests import datagen
from deepac.builtin_loading import BuiltinLoader
from deepac import __file__
from deepac.explain.tests import ExplainTester
from deepac.gwpa.tests import GWPATester
import configparser
import os


class Tester:

    """
    Tester class.

    """

    def __init__(self, n_cpus=8, n_gpus=0, builtin_configs=None, builtin_weights=None,
                 explain=False, gwpa=False, do_all=False, do_quick=False, keep=False):
        self.n_cpus = n_cpus
        self.n_gpus = n_gpus
        self.builtin_configs = builtin_configs
        self.builtin_weights = builtin_weights
        self.bloader = BuiltinLoader(builtin_configs, builtin_weights)
        self.explain = explain
        self.gwpa = gwpa
        self.do_all = do_all
        self.do_quick = do_quick
        self.keep = keep

    def run_tests(self):
        """Generate sample data and run all tests."""
        if not self.keep and os.path.exists("deepac-tests"):
            print("Deleting previous test output...")
            for root, dirs, files in os.walk("deepac-tests", topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
        quick = self.do_quick or self.explain or self.gwpa

        gwpatester = None

        print("TEST: Generating data...")
        datagen.generate_sample_data()
        print("TEST: Preprocessing data...")
        self.test_preproc()
        print("TEST: Training...")
        self.test_train(quick)
        print("TEST: Predicting...")
        self.test_pred(quick)
        print("TEST: Evaluating...")
        self.test_eval()
        print("TEST: Converting...")
        self.test_convert()
        print("TEST: Filtering...")
        self.test_filter()

        if self.do_all or self.gwpa:
            gwpatester = GWPATester(self.n_cpus)
            print("X-TEST: Fragmenting genomes...")
            gwpatester.test_fragment()
            print("X-TEST: Genome-wide phenotype potential map...")
            gwpatester.test_genomemap()
            print("X-TEST: Gene ranking...")
            gwpatester.test_granking()
            print("X-TEST: Filter activations...")
            gwpatester.test_factiv()
            print("X-TEST: Filter enrichment...")
            gwpatester.test_fenrichment()

        if self.do_all or self.explain:
            explaintester = ExplainTester(self.n_cpus, self.n_gpus)
            print("X-TEST: Maxact (DeepBind)...")
            explaintester.test_maxact()
            # SHAP
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
            print("X-TEST: transfac2IC...")
            explaintester.test_transfac2ic()
            print("X-TEST: Motif comparison...")
            explaintester.test_motif_compare()

        if self.do_all or self.gwpa:
            print("X-TEST: Nucleotide contribution map...")
            gwpatester.test_ntcontribs()

        print("TEST: OK")

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
        K.clear_session()

        if not quick:
            print("TEST: Training (rapid)...")
            paprconfig = self.bloader.get_rapid_training_config(self.n_cpus, self.n_gpus)
            self.__config_train(paprconfig).train()
            runname = paprconfig.runname
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "nn-{}-e001.h5".format(runname)))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "nn-{}-e002.h5".format(runname)))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "training-{}.csv".format(runname)))), "Training failed."
            K.clear_session()

            print("TEST: Training (sensitive)...")
            paprconfig = self.bloader.get_sensitive_training_config(self.n_cpus, self.n_gpus)
            self.__config_train(paprconfig).train()
            runname = paprconfig.runname
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "nn-{}-e001.h5".format(runname)))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "nn-{}-e002.h5".format(runname)))), "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "training-{}.csv".format(runname)))), "Training failed."
            K.clear_session()

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
            paprconfig = self.bloader.get_rapid_training_config(self.n_cpus, self.n_gpus)
            runname = paprconfig.runname
            model = self.bloader.load_rapid_model(self.n_cpus, self.n_gpus, log_path="deepac-tests")
            predict_npy(model, os.path.join("deepac-tests", "sample_val_data.npy"),
                        os.path.join("deepac-tests", "{}-logs".format(runname), "val-pred-rapid.npy"))
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "val-pred-rapid.npy"))), "Prediction failed."

            print("TEST: Predicting (sensitive)...")
            paprconfig = self.bloader.get_sensitive_training_config(self.n_cpus, self.n_gpus)
            runname = paprconfig.runname
            model = self.bloader.load_sensitive_model(self.n_cpus, self.n_gpus, log_path="deepac-tests")
            predict_npy(model, os.path.join("deepac-tests", "sample_val_data.npy"),
                        os.path.join("deepac-tests", "{}-logs".format(runname), "val-pred-sensitive.npy"))
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
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
                      os.path.join("deepac-tests", "deepac-test-logs",
                                   "deepac-test-e002-predictions-sample_val-pos.npy"))
        filter_fasta(os.path.join("deepac-tests", "sample-val-pos.fasta"),
                     os.path.join("deepac-tests", "deepac-test-logs",
                                  "deepac-test-e002-predictions-sample_val-pos.npy"),
                     os.path.join("deepac-tests", "sample-val-pos-filtered.fasta"))
        assert (os.path.isfile(os.path.join("deepac-tests", "sample-val-pos-filtered.fasta"))), "Filtering failed."
