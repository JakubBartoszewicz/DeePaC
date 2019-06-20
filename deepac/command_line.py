"""@package deepac.command_line
A DeePaC CLI. Support subcommands, prediction with built-in and custom models, training, evaluation, data preprocessing.

"""
import numpy as np
import tensorflow as tf
import random as rn

import argparse
import configparser
import os
import keras.backend as K
from keras.models import load_model

from deepac.predict import predict_fasta, predict_npy, filter_fasta
from deepac.nn_train import RCNet, RCConfig
from deepac.preproc import preproc
from deepac.eval.eval import evaluate_reads
from deepac.eval.eval_species import evaluate_species
from deepac.eval.eval_ens import evaluate_ensemble
from deepac.convert import convert_cudnn
from deepac import builtin_loading
from deepac.tests import testcalls
from deepac import __version__


def main():
    """Run DeePaC CLI."""
    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)
    rn.seed(seed)

    parse()


def parse():
    """Parse DeePaC CLI arguments."""
    parser = argparse.ArgumentParser(prog='deepac', description="Predicting pathogenic potentials of novel DNA "
                                                                "with reverse-complement neural networks.")
    parser.add_argument('-v', '--version', dest='version', action='store_true', help='Print version.')
    subparsers = parser.add_subparsers(help='DeePaC subcommands. See command --help for details.')

    # create the parser for the "predict" command
    parser_predict = subparsers.add_parser('predict', help='Predict using a trained model.')
    parser_predict.add_argument('input', help="Input file path [.fasta].")
    parser_predict.add_argument('-a', '--array', dest='array', action='store_true', help='Use .npy input instead.')
    predict_group = parser_predict.add_mutually_exclusive_group(required=True)
    predict_group.add_argument('-s', '--sensitive', dest='sensitive', action='store_true',
                               help='Use the sensitive LSTM model.')
    predict_group.add_argument('-r', '--rapid', dest='rapid', action='store_true', help='Use the rapid CNN model.')
    predict_group.add_argument('-c', '--custom', dest='custom', help='Use the user-supplied, already compiled CUSTOM'
                                                                    ' model.')
    parser_predict.add_argument('-o', '--output', help="Output file path [.npy].")
    parser_predict.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores.", default=8, type=int)
    parser_predict.add_argument('-g', '--n-gpus', dest="n_gpus", help="Number of GPUs.", default=0, type=int)
    parser_predict.add_argument('-d', '--device-prefix', dest="d_pref", help="GPU name prefix.", default="/device:GPU:")
    parser_predict.set_defaults(func=run_predict)

    # create the parser for the "filter" command
    parser_filter = subparsers.add_parser('filter', help='Filter prediction results.')

    parser_filter.add_argument('input',  help="Input file path [.fasta].")
    parser_filter.add_argument('predictions', help="Predictions in matching order [.npy].")
    parser_filter.add_argument('-t', '--threshold', help="Threshold [default=0.5].", default=0.5, type=float)
    parser_filter.add_argument('-p', '--potentials', help="Print pathogenic potential values in .fasta headers.",
                               default=False, action="store_true")
    parser_filter.add_argument('--precision', help="Format pathogenic potentials to given precision "
                               "[default=3].", default=3, type=int)
    parser_filter.add_argument('-o', '--output', help="Output file path [.fasta].")
    parser_filter.set_defaults(func=run_filter)

    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train', help='Train a new model.')
    train_group = parser_train.add_mutually_exclusive_group(required=True)
    train_group.add_argument('-s', '--sensitive', dest='sensitive', action='store_true',
                               help='Use the sensitive LSTM model.')
    train_group.add_argument('-r', '--rapid', dest='rapid', action='store_true', help='Use the rapid CNN model.')
    train_group.add_argument('-c', '--custom', dest='custom', help='Use the user-supplied configuration file.')
    parser_train.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores.", default=8, type=int)
    parser_train.add_argument('-g', '--n-gpus', dest="n_gpus", help="Number of GPUs.", default=1, type=int)
    parser_train.add_argument('-d', '--device-prefix', dest="d_pref", help="GPU name prefix.", default="/device:GPU:")
    parser_train.add_argument('-T', '--train-data', dest="train_data", help="Path to training data.")
    parser_train.add_argument('-t', '--train-labels', dest="train_labels", help="Path to training labels.")
    parser_train.add_argument('-V', '--val-data', dest="val_data", help="Path to validation data.")
    parser_train.add_argument('-v', '--val-labels', dest="val_labels", help="Path to validation labels.")
    parser_train.set_defaults(func=run_train)

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
    parser_convert = subparsers.add_parser('convert', help='Convert a CuDNNLSTM to a CPU-compatible LSTM.')
    parser_convert.add_argument('config', help='Training config file.')
    parser_convert.add_argument('model', help='Saved model.')
    parser_convert.add_argument('-w', '--weights', dest='from_weights', help="Use prepared weights instead of the "
                                                                             "model file.", action="store_true")
    parser_convert.set_defaults(func=run_convert)

    parser_test = subparsers.add_parser('test', help='Run additional tests.')
    parser_test.add_argument('-n', '--n-cpus', dest="n_cpus", help="Number of CPU cores.", default=8, type=int)
    parser_test.add_argument('-g', '--n-gpus', dest="n_gpus", help="Number of GPUs.", default=0, type=int)
    parser_test.set_defaults(func=run_tests)

    args = parser.parse_args()

    if args.version:
        print(__version__)
    elif hasattr(args, 'func'):
        args.func(args)
    else:
        print(__version__)
        parser.print_help()


def run_train(args):
    """Parse the config file and train the NN on Illumina reads."""
    print("Using {} GPUs.".format(args.n_gpus))
    if args.sensitive:
        paprconfig = builtin_loading.get_sensitive_training_config(args.n_cpus, args.n_gpus, d_pref=args.d_pref)
    elif args.rapid:
        paprconfig = builtin_loading.get_rapid_training_config(args.n_cpus, args.n_gpus, d_pref=args.d_pref)
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

    if K.backend() == 'tensorflow':
        paprconfig.set_tf_session()
    paprnet = RCNet(paprconfig)
    paprnet.load_data()
    paprnet.compile_model()
    paprnet.train()


def run_predict(args):
    """Predict pathogenic potentials from a fasta/npy file."""
    print("Using {} GPUs.".format(args.n_gpus))
    if args.n_cpus <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid number of cores" % args.n_cpus)
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_predictions.npy"

    if args.sensitive:
        model = builtin_loading.load_sensitive_model(args.n_cpus, args.n_gpus, d_pref=args.d_pref)
    elif args.rapid:
        model = builtin_loading.load_rapid_model(args.n_cpus, args.n_gpus, d_pref=args.d_pref)
    else:
        model = load_model(args.custom)

    if args.array:
        predict_npy(model, args.input, args.output)
    else:
        predict_fasta(model, args.input, args.output, args.n_cpus)


def run_filter(args):
    """Filter a reads in a fasta file by pathogenic potential."""
    if args.precision <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid precision value" % args.precision)
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_filtered_{}.fasta".format(args.threshold)
    filter_fasta(args.input, args.predictions, args.output, args.threshold, args.potentials, args.precision)


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
    """Convert a CuDNNLSTM to a CPU-compatible LSTM."""
    config = configparser.ConfigParser()
    config.read(args.config)
    convert_cudnn(config, args.model, args.from_weights)

def run_tests(args):
    """Run tests."""
    testcalls.run_tests(args.n_cpus, args.n_gpus)

if __name__ == "__main__":
    main()
