import argparse
import configparser
import keras.backend as K
from keras.models import load_model
import os

from deepac.predict import predict, filter
from deepac.nn_train import RCConfig, RCNet
from deepac.preproc import preproc
from deepac.eval.eval import evaluate_reads
from deepac.eval.eval_species import evaluate_species
from deepac.eval.eval_ens import evaluate_ensemble
from deepac.convert import convert_cudnn
from deepac import __file__


def main():
    args = parse()
    args.func(args)


def parse():
    parser = argparse.ArgumentParser(prog='deepac', description="Predicting pathogenic potentials of novel DNA "
                                                                "with reverse-complement neural networks.")
    subparsers = parser.add_subparsers(help='DeePaC subcommands. See command --help for details.')

    # create the parser for the "predict" command
    parser_predict = subparsers.add_parser('predict', help='Predict using a trained model.')
    parser_predict.add_argument("input", help="Input file path [.fasta]")
    predict_group = parser_predict.add_mutually_exclusive_group(required=True)
    predict_group.add_argument('-s, --sensitive', dest='sensitive', action='store_true',
                               help='Use the sensitive LSTM model.')
    predict_group.add_argument('-r, --rapid', dest='rapid', action='store_true', help='Use the rapid CNN model.')
    predict_group.add_argument('-p, --patric-strain', dest='strain', action='store_true',
                               help='Use the strain-based CNN model trained on PATRIC data.')
    predict_group.add_argument('-c, --custom', dest='custom', help='Use the user-supplied, already compiled CUSTOM'
                                                                   ' model')
    parser_predict.add_argument("-o", "--output", help="Output file path [.npy]")
    parser_predict.add_argument("-n", "--n-cpus", dest="n_cpus", help="Number of CPU cores", default=8, type=int)
    parser_predict.add_argument("-g", "--n-gpus", dest="n_gpus", help="Number of GPUs", default=1, type=int)
    parser_predict.set_defaults(func=run_predict)

    # create the parser for the "filter" command
    parser_filter = subparsers.add_parser('filter', help='Filter using a trained model.')

    parser_filter.add_argument("input",  help="Input file path [.fasta]")
    parser_filter.add_argument("predictions", help="Predictions in matching order [.npy]")
    parser_filter.add_argument("-t", "--threshold", help="Threshold [default=0.5]", default=0.5, type=float)
    parser_filter.add_argument("-p", "--potentials", help="Print pathogenic potential values in .fasta headers",
                               default=False, action="store_true")
    parser_filter.add_argument("--precision", help="Format pathogenic potentials to given precision "
                               "[default=3]", default=3, type=int)
    parser_filter.add_argument("-o", "--output", help="Output file path [.fasta]")
    parser_filter.set_defaults(func=run_filter)

    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train', help='Train a new model.')
    parser_train.add_argument('config', help='Training config file.')
    parser_train.set_defaults(func=run_train)

    # create the parser for the "preproc" command
    parser_preproc = subparsers.add_parser('preproc', help='Convert fasta files to numpy arrays for training.')
    parser_preproc.add_argument('config', type=int, help='Preprocessing config file.')
    parser_preproc.set_defaults(func=run_preproc)

    # create the parser for the "eval" command
    parser_eval = subparsers.add_parser('eval', help='Predict using a trained model.')
    eval_group = parser_eval.add_mutually_exclusive_group(required=True)
    eval_group.add_argument('-s, --species', dest='species_config', help='Species-wise evaluation.')
    eval_group.add_argument('-r, --reads', dest='reads_config', help='Read-wise evaluation.')
    eval_group.add_argument('-e, --ensemble', dest='ens_config', help='Simple ensemble evaluation.')
    parser_eval.set_defaults(func=run_evaluate)

    # create the parser for the "convert" command
    parser_convert = subparsers.add_parser('convert', help='Convert a CuDNNLSTM to a CPU-compatible LSTM.')
    parser_convert.add_argument("config", help='Training config file.')
    parser_convert.add_argument("model", help='Saved model.')
    parser_convert.add_argument("--no_prep", help="Use prepared weights instead of the model file", action="store_true")
    parser_convert.set_defaults(func=run_convert)

    args = parser.parse_args()
    return args


def run_train(args):
    """Parse the config file and train the NN on Illumina reads."""
    config = configparser.ConfigParser()
    config.read(args.config)
    paprconfig = RCConfig(config)
    if K.backend() == 'tensorflow':
        paprconfig.set_tf_session()
    paprnet = RCNet(paprconfig)
    paprnet.load_data()
    paprnet.compile_model()
    paprnet.train()


def run_predict(args):
    if args.n_cpus <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid number of cores" % args.n_cpus)
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_predictions.npy"

    if args.custom is not None:
        model = load_model(args.custom)
    elif args.sensitive:
        model = load_sensitive_model(args.n_cpus, args.n_gpus)
    elif args.rapid:
        model = load_rapid_model(args.n_cpus, args.n_gpus)
    elif args.strain:
        model = load_strain_model(args.n_cpus, args.n_gpus)

    predict(model, args.input, args.output, args.n_cpus)


def run_filter(args):
    if args.precision <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid precision value" % args.precision)
    if args.output is None:
        args.output = os.path.splitext(args.input)[0] + "_filtered_{}.fasta".format(args.threshold)
    filter(args.input, args.predictions, args.output, args.threshold, args.potentials, args.precision)


def run_preproc(args):
    """Parse the config file and preprocess the Illumina reads."""
    config = configparser.ConfigParser()
    config.read(args.config)
    preproc(config)


def run_evaluate(args):
    """Parse the config file and evaluate the NN on Illumina reads."""
    if args.species_config is not None:
        evaluate_species(args.species_config)
    elif args.reads_config is not None:
        evaluate_reads(args.reads_config)
    elif args.ens_config is not None:
        evaluate_ensemble(args.ens_config)


def run_convert(args):
    """Convert a CuDNNLSTM to a CPU-compatible LSTM."""
    config = configparser.ConfigParser()
    config.read(args.config_file)
    convert_cudnn(config, args.saved_model, args.no_prep)


def load_sensitive_model(n_cpus, n_gpus, device_parallel=False):
    if n_gpus > 0:
        device_parallel = True
    return load_builtin_model("nn-img-sensitive-lstm", n_cpus, n_gpus, device_parallel)


def load_rapid_model(n_cpus, n_gpus, device_parallel=False):
    return load_builtin_model("nn-img-rapid-cnn", n_cpus, n_gpus, device_parallel)


def load_strain_model(n_cpus, n_gpus, device_parallel=False):
    return load_builtin_model("nn-patric-strain-cnn-bn", n_cpus, n_gpus, device_parallel)


def load_builtin_model(prefix, n_cpus, n_gpus, device_parallel):
    config_path = os.path.join(os.path.dirname(__file__), "builtin", "config", "{}.ini".format(prefix))
    weights_path = os.path.join(os.path.dirname(__file__), "builtin", "weights", "{}.h5".format(prefix))
    config = configparser.ConfigParser()
    config.read(config_path)
    paprconfig = RCConfig(config)
    paprconfig.n_cpus = n_cpus
    paprconfig.n_gpus = n_gpus
    paprconfig.device_parallel = device_parallel

    if K.backend() == 'tensorflow':
        paprconfig.set_tf_session()

    paprnet = RCNet(paprconfig)

    paprnet.model.load_weights(weights_path)

    return paprnet.model


if __name__ == "__main__":
    main()
