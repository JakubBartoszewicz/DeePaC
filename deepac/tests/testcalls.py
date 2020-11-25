import tensorflow as tf
from deepac.predict import predict_fasta, filter_fasta
from deepac.nn_train import RCConfig, RCNet
from deepac.eval.eval import evaluate_reads
from deepac.eval.eval_ens import evaluate_ensemble
from deepac.convert import convert_cudnn
from deepac import preproc
from deepac.tests import datagen
from deepac.tests.rctest import compare_rc
from deepac.builtin_loading import BuiltinLoader
from deepac import __file__
from deepac.explain.tests import ExplainTester
from deepac.gwpa.tests import GWPATester
from multiprocessing import Process
import configparser
import os


class Tester:

    """
    Tester class.

    """

    def __init__(self, n_cpus=8, builtin_configs=None, builtin_weights=None, explain=False, gwpa=False, do_all=False,
                 do_quick=False, keep=False, scale=1, tpu_resolver=None, input_modes=None, additivity_check=False,
                 large=False):
        self.n_cpus = n_cpus
        self.builtin_configs = builtin_configs
        self.builtin_weights = builtin_weights
        self.bloader = BuiltinLoader(builtin_configs, builtin_weights)
        self.explain = explain
        self.gwpa = gwpa
        self.do_all = do_all
        self.do_quick = do_quick
        self.keep = keep
        self.scale = scale
        self.tpu_resolver = tpu_resolver
        self.additivity_check = additivity_check
        self.do_large = large
        self.test_config = "nn-test-L.ini" if self.do_large else "nn-test.ini"
        # all by default, unless using a TPU when it defaults to memory
        self.input_modes = ["memory"] if tpu_resolver is not None and input_modes is None else input_modes
        # all are true by default, unless input_modes is specified
        do_sequence_input = self.input_modes is None or "sequence" in input_modes
        if float(tf.__version__[:3]) == 2.2:
            print('Disabling keras Sequence input in TF 2.2.')
            do_sequence_input = False
        self.input_modes_dict = {"memory": self.input_modes is None or "memory" in input_modes,
                                 "sequence": do_sequence_input,
                                 "tfdata": self.input_modes is None or "tfdata" in input_modes}

    def run_datagen(self, npy=True, tfrec=True):
        print("TEST: Generating data...")
        datagen.generate_sample_data(n_train=1024 * self.scale, n_val=1024 * self.scale)
        print("TEST: Preprocessing data...")
        self.test_preproc(npy, tfrec)

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
        if not self.keep:
            self.run_datagen(npy=True,
                             tfrec=self.input_modes_dict["tfdata"])
        print("TEST: Training...")
        self.test_train(quick)
        print("TEST: Predicting...")
        self.test_pred(quick)
        print("TEST: Evaluating...")
        self.test_eval()
        print("TEST: Converting...")
        self.test_convert()
        print("TEST: Continuing training...")
        self.test_train(quick=True, epoch_start=2, epoch_end=4)
        print("TEST: Filtering...")
        self.test_filter()

        if self.do_all or self.gwpa:
            gwpatester = GWPATester(self.n_cpus, self.additivity_check)
            print("X-TEST: gff2genome...")
            gwpatester.test_gff2genome()
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
            explaintester = ExplainTester(self.n_cpus, self.additivity_check)
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

    def test_preproc(self, npy=True, tfrec=True):
        """Test preprocessing."""
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "preproc-train.ini"))
        if npy:
            config['Options']['Use_TFData'] = "False"
            preproc.preproc(config)
            assert (os.path.isfile(os.path.join("deepac-tests", "sample_train_data.npy"))), "Preprocessing failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "sample_train_labels.npy"))), "Preprocessing failed."
        if tfrec:
            config['Options']['Use_TFData'] = "True"
            config['Options']['Do_shuffle'] = "True"
            p = Process(target=preproc.preproc, args=(config,))
            p.start()
            p.join()
            assert (os.path.isfile(os.path.join("deepac-tests", "sample_train_data",
                                                "sample_train_data_0-{}.tfrec".format(128*self.scale-1)))),\
                "TFData Preprocessing failed."

        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "preproc-val.ini"))
        if npy:
            config['Options']['Use_TFData'] = "False"
            preproc.preproc(config)
            assert (os.path.isfile(os.path.join("deepac-tests", "sample_val_data.npy"))), "Preprocessing failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "sample_val_labels.npy"))), "Preprocessing failed."
        if tfrec:
            config['Options']['Use_TFData'] = "True"
            config['Options']['Do_shuffle'] = "True"
            p = Process(target=preproc.preproc, args=(config,))
            p.start()
            p.join()
            assert (os.path.isfile(os.path.join("deepac-tests", "sample_val_data",
                                                "sample_val_data_0-{}.tfrec".format(128*self.scale-1)))),\
                "TFData Preprocessing failed."

    def test_train(self, quick=False, epoch_start=0, epoch_end=2):
        """Test training."""
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", self.test_config))

        if self.input_modes_dict["memory"]:
            print("TEST: Training (custom - .npy in memory)...")
            config['DataLoad']['Use_TFData'] = "False"
            config['DataLoad']['LoadTrainingByBatch'] = "False"
            paprconfig = RCConfig(config)
            self.__config_train(paprconfig, epoch_start, epoch_end).train()
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "deepac-test-e{epoch:03d}.h5".format(epoch=epoch_start+1)))),\
                "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "deepac-test-e{epoch:03d}.h5".format(epoch=epoch_end)))),\
                "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "training-deepac-test.csv"))), "Training failed."

        if self.input_modes_dict["sequence"]:
            print("TEST: Training (custom - keras sequence)...")
            config['DataLoad']['Use_TFData'] = "False"
            config['DataLoad']['LoadTrainingByBatch'] = "True"
            paprconfig = RCConfig(config)
            self.__config_train(paprconfig, epoch_start, epoch_end).train()
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "deepac-test-e{epoch:03d}.h5".format(epoch=epoch_start+1)))),\
                "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "deepac-test-e{epoch:03d}.h5".format(epoch=epoch_end)))),\
                "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "training-deepac-test.csv"))), "Training failed."

        if self.input_modes_dict["tfdata"]:
            print("TEST: Training (custom - tfrecord)...")
            config['DataLoad']['Use_TFData'] = "True"
            paprconfig = RCConfig(config)
            self.__config_train(paprconfig, epoch_start, epoch_end).train()
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "deepac-test-e{epoch:03d}.h5".format(epoch=epoch_start+1)))),\
                "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "deepac-test-e{epoch:03d}.h5".format(epoch=epoch_end)))),\
                "Training failed."
            assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                                "training-deepac-test.csv"))), "Training failed."

        if not quick:
            if self.input_modes_dict["memory"]:
                print("TEST: Training (rapid - .npy in memory)...")
                paprconfig = self.bloader.get_rapid_training_config()
                paprconfig.use_tf_data = False
                paprconfig.use_generators_keras = False
                self.__config_train(paprconfig, epoch_start, epoch_end).train()
                runname = paprconfig.runname
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_start+1)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_end)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                    "training-{}.csv".format(runname)))), "Training failed."
            if self.input_modes_dict["sequence"]:
                print("TEST: Training (rapid - keras sequence)...")
                paprconfig = self.bloader.get_rapid_training_config()
                paprconfig.use_tf_data = False
                paprconfig.use_generators_keras = True
                self.__config_train(paprconfig, epoch_start, epoch_end).train()
                runname = paprconfig.runname
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_start+1)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_end)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                    "training-{}.csv".format(runname)))), "Training failed."

            if self.input_modes_dict["tfdata"]:
                print("TEST: Training (rapid - tfrecord)...")
                paprconfig = self.bloader.get_rapid_training_config()
                paprconfig.use_tf_data = True
                self.__config_train(paprconfig, epoch_start, epoch_end).train()
                runname = paprconfig.runname
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_start+1)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_end)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                    "training-{}.csv".format(runname)))), "Training failed."

            if self.input_modes_dict["memory"]:
                print("TEST: Training (sensitive - .npy in memory)...")
                paprconfig = self.bloader.get_sensitive_training_config()
                paprconfig.use_tf_data = False
                paprconfig.use_generators_keras = False
                self.__config_train(paprconfig, epoch_start, epoch_end).train()
                runname = paprconfig.runname
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_start+1)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_end)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                    "training-{}.csv".format(runname)))), "Training failed."

            if self.input_modes_dict["sequence"]:
                print("TEST: Training (sensitive - keras sequence)...")
                paprconfig = self.bloader.get_sensitive_training_config()
                paprconfig.use_tf_data = False
                paprconfig.use_generators_keras = True
                self.__config_train(paprconfig, epoch_start, epoch_end).train()
                runname = paprconfig.runname
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_start+1)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_end)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                    "training-{}.csv".format(runname)))), "Training failed."

            if self.input_modes_dict["tfdata"]:
                print("TEST: Training (sensitive - tfrecord)...")
                paprconfig = self.bloader.get_sensitive_training_config()
                paprconfig.use_tf_data = True
                self.__config_train(paprconfig, epoch_start, epoch_end).train()
                runname = paprconfig.runname
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_start+1)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{runname}-logs".format(runname=runname),
                                                    "{runname}-e{epoch:03d}.h5".format(runname=runname,
                                                                                       epoch=epoch_end)))),\
                    "Training failed."
                assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                    "training-{}.csv".format(runname)))), "Training failed."

    def __config_train(self, paprconfig, epoch_start=0, epoch_end=2):
        """Set sample data paths and compile."""
        if paprconfig.use_tf_data and tf.executing_eagerly():
            paprconfig.x_train_path = os.path.join("deepac-tests", "sample_train_data")
            paprconfig.y_train_path = os.path.join("deepac-tests", "sample_train_labels")
            paprconfig.x_val_path = os.path.join("deepac-tests", "sample_val_data")
            paprconfig.y_val_path = os.path.join("deepac-tests", "sample_val_labels")
        else:
            paprconfig.x_train_path = os.path.join("deepac-tests", "sample_train_data.npy")
            paprconfig.y_train_path = os.path.join("deepac-tests", "sample_train_labels.npy")
            paprconfig.x_val_path = os.path.join("deepac-tests", "sample_val_data.npy")
            paprconfig.y_val_path = os.path.join("deepac-tests", "sample_val_labels.npy")
        paprconfig.epoch_start = epoch_start
        paprconfig.epoch_end = epoch_end
        paprconfig.log_superpath = "deepac-tests"
        paprconfig.log_dir = paprconfig.log_superpath + "/{runname}-logs".format(runname=paprconfig.runname)
        paprconfig.set_tpu_resolver(self.tpu_resolver)

        paprnet = RCNet(paprconfig)
        paprnet.load_data()
        paprnet.compile_model()
        return paprnet

    def test_pred(self, quick=False):
        """Test predicting."""
        print("TEST: Predicting (custom)...")
        model = tf.keras.models.load_model(os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002.h5"))
        compare_rc(model, os.path.join("deepac-tests", "sample_val_data.npy"),
                   os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002-predictions-sample_val.npy"),
                   replicates=1)
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test-e002-predictions-sample_val.npy"))), "Prediction failed."

        if not quick:
            print("TEST: Predicting (rapid)...")
            paprconfig = self.bloader.get_rapid_training_config()
            runname = paprconfig.runname
            model = self.bloader.load_rapid_model(log_path="deepac-tests", tpu_resolver=self.tpu_resolver)
            compare_rc(model, os.path.join("deepac-tests", "sample_val_data.npy"),
                       os.path.join("deepac-tests", "{}-logs".format(runname), "val-pred-rapid.npy"))
            assert (os.path.isfile(os.path.join("deepac-tests", "{}-logs".format(runname),
                                                "val-pred-rapid.npy"))), "Prediction failed."

            print("TEST: Predicting (sensitive)...")
            paprconfig = self.bloader.get_sensitive_training_config()
            runname = paprconfig.runname
            model = self.bloader.load_sensitive_model(log_path="deepac-tests", tpu_resolver=self.tpu_resolver)
            compare_rc(model, os.path.join("deepac-tests", "sample_val_data.npy"),
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

        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", "eval_ens-test.ini"))
        evaluate_ensemble(config)
        assert (os.path.isfile(os.path.join("deepac-tests",
                                            "ens01-metrics.csv"))), "Evaluation failed."
        assert (os.path.isfile(os.path.join("deepac-tests",
                                            "ens01_sample_val_auc.png"))), "Evaluation failed."
        assert (os.path.isfile(os.path.join("deepac-tests",
                                            "ens01_sample_val_aupr.png"))), "Evaluation failed."

    def test_convert(self):
        """Test converting."""
        config = configparser.ConfigParser()
        config.read(os.path.join(os.path.dirname(__file__), "tests", "configs", self.test_config))
        config['Devices']['DistStrategy'] = "OneDeviceStrategy"
        config['Devices']['Device_build'] = "CPU:0"
        convert_cudnn(config, os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002.h5"), False)
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test-e002_converted.h5"))), "Conversion failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test-e002_weights.h5"))), "Conversion failed."
        config['Architecture']['MC_Dropout'] = 'True'
        convert_cudnn(config, os.path.join("deepac-tests", "deepac-test-logs", "deepac-test-e002_converted.h5"), False)
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test-e002_converted_converted.h5"))), "Conversion failed."
        assert (os.path.isfile(os.path.join("deepac-tests", "deepac-test-logs",
                                            "deepac-test-e002_converted_weights.h5"))), "Conversion failed."

    def test_filter(self):
        """Test filtering."""
        model = tf.keras.models.load_model(os.path.join("deepac-tests", "deepac-test-logs",
                                                        "deepac-test-e002_converted_converted.h5"))
        predict_fasta(model, os.path.join("deepac-tests", "sample-test.fasta"),
                      os.path.join("deepac-tests", "deepac-test-logs",
                                   "deepac-test-e002-predictions-sample_test.npy"),
                      replicates=5)
        filter_fasta(os.path.join("deepac-tests", "sample-test.fasta"),
                     os.path.join("deepac-tests", "deepac-test-logs",
                                  "deepac-test-e002-predictions-sample_test.npy"),
                     os.path.join("deepac-tests", "sample-test-filtered-pos.fasta"),
                     print_potentials=True,
                     output_neg=os.path.join("deepac-tests", "sample-test-filtered-neg.fasta"),
                     confidence_thresh=0.65,
                     output_undef=os.path.join("deepac-tests", "sample-test-filtered-undef.fasta"),
                     pred_uncertainty=os.path.join("deepac-tests", "deepac-test-logs",
                                                   "deepac-test-e002-predictions-sample_test-std.npy"))
        assert (os.path.isfile(os.path.join("deepac-tests", "sample-test-filtered-pos.fasta"))), "Filtering failed."
