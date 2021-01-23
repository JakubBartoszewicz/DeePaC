"""@package deepac.command_line
A DeePaC CLI. Support subcommands, prediction with built-in and custom models, training, evaluation, data preprocessing.

"""
import sklearn # to load libgomp early to solve problems with static TLS on some systems like bioconda mulled tests
import matplotlib.pyplot as plt # also to solve import ordering problems in bioconda mulled tests
import numpy as np
import tensorflow as tf
import random as rn
import argparse
import configparser
import os
import shutil
import multiprocessing

from deepac.predict import predict_fasta, predict_npy, filter_fasta
from deepac.nn_train import RCNet, RCConfig
from deepac.preproc import preproc
from deepac.eval.eval import evaluate_reads
from deepac.eval.eval_species import evaluate_species
from deepac.eval.eval_ens import evaluate_ensemble
from deepac.convert import convert_cudnn
from deepac.builtin_loading import BuiltinLoader, RemoteLoader
from deepac.tests.testcalls import Tester
from deepac.tests.rctest import compare_rc
from deepac import __version__
from deepac import __file__
from deepac.utils import config_gpus, config_cpus, config_tpus
from deepac.explain.command_line import add_explain_parser
from deepac.gwpa.command_line import add_gwpa_parser


def main():
    """Run DeePaC CLI."""
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)
    rn.seed(seed)
    modulepath = os.path.dirname(__file__)
    builtin_configs = {"rapid": os.path.join(modulepath, "builtin", "config", "nn-img-rapid-cnn.ini"),
                       "sensitive": os.path.join(modulepath, "builtin", "config", "nn-img-sensitive-lstm.ini")}
    builtin_weights = {"rapid": os.path.join(modulepath, "builtin", "weights", "nn-img-rapid-cnn.h5"),
                       "sensitive": os.path.join(modulepath, "builtin", "weights", "nn-img-sensitive-lstm.h5")}
    remote_repo_url = "https://zenodo.org/api/records/4456008"
    runner = MainRunner(builtin_configs, builtin_weights, remote_repo_url)
    runner.parse()


def run_filter(args):
    """Filter a reads in a fasta file by pathogenic potential."""
    if args.precision <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid precision value" % args.precision)
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_filtered_{}.fasta".format(args.threshold)
    filter_fasta(args.input, args.predictions, args.output, args.threshold, args.potentials, args.precision,
                 pred_uncertainty=args.std)


def run_preproc(args):
    """Parse the config file and preprocess the Illumina reads."""
    config = configparser.ConfigParser()
    config.read(args.config)
    preproc(config)


def run_evaluate(args):
    """Parse the config file and evaluate the NN on Illumina reads."""
    config = configparser.ConfigParser()
    if args.species_config is not None:
        config.read(args.species_config)
        evaluate_species(config)
    elif args.reads_config is not None:
        config.read(args.reads_config)
        evaluate_reads(config)
    elif args.ens_config is not None:
        config.read(args.ens_config)
        evaluate_ensemble(config)


def run_convert(args):
    """Rebuild the network using a modified configuration."""
    config = configparser.ConfigParser()
    config.read(args.config)
    convert_cudnn(config, args.model, args.from_weights, args.init)


def run_templates(args):
    """Get config templates (in this directory)."""
    modulepath = os.path.dirname(__file__)
    extra_templates_path = os.path.join(modulepath, "builtin", "config_templates")
    training_templates_path = os.path.join(modulepath, "builtin", "config")
    shutil.copytree(training_templates_path, os.path.join(os.getcwd(), "deepac_training_configs"))
    shutil.copytree(extra_templates_path, os.path.join(os.getcwd(), "deepac_extra_configs"))


def global_setup(args):
    tpu_resolver = None
    if args.tpu:
        tpu_resolver = config_tpus(args.tpu)
    if args.no_eager:
        print("Disabling eager mode...")
        tf.compat.v1.disable_v2_behavior()
    if args.debug_device:
        tf.debugging.set_log_device_placement(True)
    if args.force_cpu:
        tf.config.set_visible_devices([], 'GPU')
        args.gpus = None
    default_verbosity = '3' if args.subparser == 'test' else '2'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.debug_tf) if args.debug_tf is not None else default_verbosity

    return tpu_resolver


def add_global_parser(gparser):
    gparser.add_argument('-v', '--version', dest='version', action='store_true', help='Print version.')
    gparser.add_argument('--debug-no-eager', dest="no_eager", help="Disable eager mode.",
                         default=False, action="store_true")
    gparser.add_argument('--debug-tf', dest="debug_tf", help="Set tensorflow debug info verbosity level. "
                                                             "0 = max, 3 = min. Default: 2 (errors);"
                                                             " 3 for tests (muted)", type=int)
    gparser.add_argument('--debug-device', dest="debug_device", help="Enable verbose device placement information.",
                         default=False, action="store_true")
    gparser.add_argument('--force-cpu', dest="force_cpu", help="Use a CPU even if GPUs are available.",
                         default=False, action="store_true")
    gparser.add_argument('--tpu', help="TPU name: 'colab' for Google Colab, or name of your TPU on GCE.")

    return gparser


class MainRunner:
    def __init__(self, builtin_configs=None, builtin_weights=None, remote_repo_url=None):
        self.builtin_configs = builtin_configs
        self.builtin_weights = builtin_weights
        self.bloader = BuiltinLoader(self.builtin_configs, self.builtin_weights)
        self.rloader = RemoteLoader(remote_repo_url)
        self.tpu_resolver = None

    def run_train(self, args):
        """Parse the config file and train the NN on Illumina reads."""
        if args.tpu is None:
            config_cpus(args.n_cpus)
            config_gpus(args.gpus)
        if args.sensitive:
            paprconfig = self.bloader.get_sensitive_training_config()
        elif args.rapid:
            paprconfig = self.bloader.get_rapid_training_config()
        else:
            config = configparser.ConfigParser()
            config.read(args.custom)
            paprconfig = RCConfig(config)

        if args.train_data:
            paprconfig.x_train_path = args.train_data
        if args.train_labels:
            paprconfig.y_train_path = args.train_labels
        if args.val_data:
            paprconfig.x_val_path = args.val_data
        if args.val_labels:
            paprconfig.y_val_path = args.val_labels
        if args.run_name:
            paprconfig.runname = args.run_name
            paprconfig.log_dir = os.path.join(paprconfig.log_superpath,
                                              "{runname}-logs".format(runname=paprconfig.runname))

        paprconfig.set_tpu_resolver(self.tpu_resolver)
        paprnet = RCNet(paprconfig)
        paprnet.load_data()
        paprnet.compile_model()
        paprnet.train()

    def run_predict(self, args):
        """Predict pathogenic potentials from a fasta/npy file."""
        if args.tpu is None:
            config_cpus(args.n_cpus)
            config_gpus(args.gpus)
        if args.output is None:
            args.output = os.path.splitext(args.input)[0] + "_predictions.npy"

        if args.sensitive:
            model = self.bloader.load_sensitive_model(training_mode=False, tpu_resolver=self.tpu_resolver)
        elif args.rapid:
            model = self.bloader.load_rapid_model(training_mode=False, tpu_resolver=self.tpu_resolver)
        else:
            if self.tpu_resolver is not None:
                tpu_strategy = tf.distribute.experimental.TPUStrategy(self.tpu_resolver)
                with tpu_strategy.scope():
                    model = tf.keras.models.load_model(args.custom)
            else:
                model = tf.keras.models.load_model(args.custom)

        if args.rc_check:
            compare_rc(model, args.input, args.output, args.plot_kind, args.alpha, replicates=args.replicates,
                       batch_size=args.batch_size)
        elif args.array:
            predict_npy(model, args.input, args.output, replicates=args.replicates, batch_size=args.batch_size)
        else:
            predict_fasta(model, args.input, args.output, args.n_cpus, replicates=args.replicates,
                          batch_size=args.batch_size)

    def run_getmodels(self, args):
        """Get built-in weights and rebuild built-in models."""
        out_dir = "deepac_builtin_models"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        modulepath = os.path.dirname(__file__)
        builtin_weights_path = os.path.join(modulepath, "builtin", "weights")
        out_weights_path = os.path.join(out_dir, "weights")
        if os.path.exists(out_weights_path):
            for root, dirs, files in os.walk(out_weights_path, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(out_weights_path)
        shutil.copytree(builtin_weights_path, out_weights_path)

        if args.sensitive:
            model = self.bloader.load_sensitive_model(training_mode=False, tpu_resolver=self.tpu_resolver)
            model.summary()
            save_path = os.path.basename(self.builtin_weights["sensitive"])
            model.save(os.path.join(out_dir, save_path))

        if args.rapid:
            model = self.bloader.load_rapid_model(training_mode=False, tpu_resolver=self.tpu_resolver)
            model.summary()
            save_path = os.path.basename(self.builtin_weights["rapid"])
            model.save(os.path.join(out_dir, save_path))

        if args.download_only or args.fetch_compile:
            self.rloader.fetch_models(out_dir, args.fetch_compile)

    def run_tests(self, args):
        """Run tests."""
        if args.tpu is None:
            n_cpus = config_cpus(args.n_cpus)
            n_gpus = config_gpus(args.gpus)
            scale = args.scale * max(1, n_gpus)
        else:
            n_cpus = multiprocessing.cpu_count()
            scale = args.scale
        tester = Tester(n_cpus, self.builtin_configs, self.builtin_weights,
                        args.explain, args.gwpa, args.all, args.quick, args.keep, scale,
                        tpu_resolver=self.tpu_resolver, input_modes=args.input_modes,
                        additivity_check=(not args.no_check), large=args.large)
        tester.run_tests()

    def parse(self):
        """Parse DeePaC CLI arguments."""
        parser = argparse.ArgumentParser(prog='deepac', description="Predicting pathogenic potentials of novel DNA "
                                                                    "with reverse-complement neural networks.")
        parser = add_global_parser(parser)
        subparsers = parser.add_subparsers(help='DeePaC subcommands. See command --help for details.', dest='subparser')

        # create the parser for the "predict" command
        parser_predict = subparsers.add_parser('predict', help='Predict using a trained model.')
        parser_predict.add_argument('input', help="Input file path [.fasta].")
        parser_predict.add_argument('-a', '--array', dest='array', action='store_true', help='Use .npy input instead.')
        predict_group = parser_predict.add_mutually_exclusive_group(required=True)
        predict_group.add_argument('-s', '--sensitive', dest='sensitive', action='store_true',
                                   help='Use the sensitive model.')
        predict_group.add_argument('-r', '--rapid', dest='rapid', action='store_true', help='Use the rapid CNN model.')
        predict_group.add_argument('-c', '--custom', dest='custom', help='Use the user-supplied, '
                                                                         'already compiled CUSTOM model.')
        parser_predict.add_argument('-o', '--output', help="Output file path [.npy].")
        parser_predict.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores. Default: all.",
                                    type=int)
        parser_predict.add_argument('-g', '--gpus', dest="gpus", nargs='+', type=int,
                                    help="GPU devices to use (comma-separated). Default: all")
        parser_predict.add_argument('-R', '--rc-check', dest="rc_check", action='store_true',
                                    help='Check RC-constraint compliance (requires .npy input).')
        parser_predict.add_argument('-b', '--batch-size', dest="batch_size", default=512, type=int,
                                    help='Batch size.')
        parser_predict.add_argument('--plot-kind', dest="plot_kind", default="scatter",
                                    help='Plot kind for the RC-constraint compliance check.')
        parser_predict.add_argument('--alpha', default=1.0, type=float,
                                    help='Alpha value for the RC-constraint compliance check plot.')
        parser_predict.add_argument('--replicates', default=1, type=int,
                                    help='Number of replicates for MC uncertainty estimation.')
        parser_predict.set_defaults(func=self.run_predict)

        # create the parser for the "filter" command
        parser_filter = subparsers.add_parser('filter', help='Filter prediction results.')

        parser_filter.add_argument('input',  help="Input file path [.fasta].")
        parser_filter.add_argument('predictions', help="Predictions in matching order [.npy].")
        parser_filter.add_argument('-t', '--threshold', help="Threshold [default=0.5].", default=0.5, type=float)
        parser_filter.add_argument('-p', '--potentials', help="Print pathogenic potential values in .fasta headers.",
                                   default=False, action="store_true")
        parser_filter.add_argument('-o', '--output', help="Output file path [.fasta].")
        parser_filter.add_argument('-s', '--std', dest="std",
                                   help="Standard deviations of predictions if MC dropout used.")
        parser_filter.add_argument('--precision', help="Format pathogenic potentials to given precision "
                                   "[default=3].", default=3, type=int)
        parser_filter.set_defaults(func=run_filter)

        # create the parser for the "train" command
        parser_train = subparsers.add_parser('train', help='Train a new model.')
        train_group = parser_train.add_mutually_exclusive_group(required=True)
        train_group.add_argument('-s', '--sensitive', dest='sensitive', action='store_true',
                                 help='Use the sensitive model.')
        train_group.add_argument('-r', '--rapid', dest='rapid', action='store_true', help='Use the rapid CNN model.')
        train_group.add_argument('-c', '--custom', dest='custom', help='Use the user-supplied configuration file.')
        parser_train.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores. Default: all.", type=int)
        parser_train.add_argument('-g', '--gpus', dest="gpus", nargs='+', type=int,
                                  help="GPU devices to use (comma-separated). Default: all")
        parser_train.add_argument('-T', '--train-data', dest="train_data", help="Path to training data.")
        parser_train.add_argument('-t', '--train-labels', dest="train_labels", help="Path to training labels.")
        parser_train.add_argument('-V', '--val-data', dest="val_data", help="Path to validation data.")
        parser_train.add_argument('-v', '--val-labels', dest="val_labels", help="Path to validation labels.")
        parser_train.add_argument('-R', '--run-name', dest="run_name", help="Run name "
                                                                            "(default: based on chosen config).")
        parser_train.set_defaults(func=self.run_train)

        # create the parser for the "preproc" command
        parser_preproc = subparsers.add_parser('preproc', help='Convert fasta files to numpy arrays for training.')
        parser_preproc.add_argument('config', help='Preprocessing config file.')
        parser_preproc.set_defaults(func=run_preproc)

        # create the parser for the "eval" command
        parser_eval = subparsers.add_parser('eval', help='Predict using a trained model.')
        eval_group = parser_eval.add_mutually_exclusive_group(required=True)
        eval_group.add_argument('-s', '--species', dest='species_config', help='Species-wise evaluation.')
        eval_group.add_argument('-r', '--reads', dest='reads_config', help='Read-wise evaluation.')
        eval_group.add_argument('-e', '--ensemble', dest='ens_config', help='Simple ensemble evaluation.')
        parser_eval.set_defaults(func=run_evaluate)

        # create the parser for the "convert" command
        parser_convert = subparsers.add_parser('convert', help='Convert and compile a model to an equivalent.')
        parser_convert.add_argument('config', help='Training config file.')
        parser_convert.add_argument('model', help='Saved model.')
        parser_convert.add_argument('-w', '--weights', dest='from_weights', action="store_true",
                                    help="Use prepared weights instead of the model file.")
        parser_convert.add_argument('-i', '--init', dest='init', action="store_true",
                                    help="Initialize a random model from config.")
        parser_convert.set_defaults(func=run_convert)

        # create the parser for the "getmodels" command
        parser_getmodel = subparsers.add_parser('getmodels', help='Get built-in weights and rebuild built-in models.')
        getmodel_group = parser_getmodel.add_argument_group()
        getmodel_group.add_argument('-s', '--sensitive', dest='sensitive', action='store_true',
                                    help='Rebuild the sensitive model.')
        getmodel_group.add_argument('-r', '--rapid', dest='rapid', action='store_true',
                                    help='Rebuild the rapid CNN model.')

        fetch_group = getmodel_group.add_mutually_exclusive_group(required=False)
        fetch_group.add_argument('-f', '--fetch', dest='fetch_compile', action='store_true',
                                 help='Fetch and compile the latest models and configs from the online repository.')
        fetch_group.add_argument('--download-only', dest='download_only', action='store_true',
                                 help='Fetch weights and config files but do not compile the models.')
        parser_getmodel.set_defaults(func=self.run_getmodels)

        parser_test = subparsers.add_parser('test', help='Run additional tests.')
        parser_test.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores. Default: all.", type=int)
        parser_test.add_argument('-g', '--gpus', dest="gpus", nargs='+', type=int,
                                 help="GPU devices to use. Default: all")
        parser_test.add_argument('-x', '--explain', dest="explain", help="Test explain workflows.",
                                 default=False, action="store_true")
        parser_test.add_argument('-p', '--gwpa', dest="gwpa", help="Test gwpa workflows.",
                                 default=False, action="store_true")
        parser_test.add_argument('-a', '--all', help="Test all functions.",
                                 default=False, action="store_true")
        parser_test.add_argument('-q', '--quick', help="Don't test heavy models (e.g. on low-memory machines"
                                                       " or when no GPU available).",
                                 default=False, action="store_true")
        parser_test.add_argument('-k', '--keep', help="Don't delete previous test output.",
                                 default=False, action="store_true")
        parser_test.add_argument('-s', '--scale', help="Generate s*1024 reads for testing (Default: s=1).",
                                 default=1, type=int)
        parser_test.add_argument('-L', '--large', help="Test a larger, more complex custom model.",
                                 default=False, action="store_true")
        parser_test.add_argument("--input-modes", nargs='*', dest="input_modes",
                                 help="Input modes to test: memory, sequence and/or tfdata. Default: all.")
        parser_test.add_argument("--no-check", dest="no_check", action="store_true",
                                 help="Disable additivity check.")
        parser_test.set_defaults(func=self.run_tests)

        parser_explain = subparsers.add_parser('explain', help='Run filter visualization workflows.')
        parser_explain = add_explain_parser(parser_explain)
        parser_explain.set_defaults(func=lambda a: parser_explain.print_help())

        parser_gwpa = subparsers.add_parser('gwpa', help='Run GWPA workflows.')
        parser_gwpa = add_gwpa_parser(parser_gwpa)
        parser_gwpa.set_defaults(func=lambda a: parser_gwpa.print_help())

        parser_templates = subparsers.add_parser('templates', help='Get config templates (in this directory).')
        parser_templates.set_defaults(func=run_templates)

        args = parser.parse_args()

        self.tpu_resolver = global_setup(args)

        if args.version:
            print(__version__)
        elif hasattr(args, 'func'):
            args.func(args)
        else:
            print(__version__)
            parser.print_help()


if __name__ == "__main__":
    main()
