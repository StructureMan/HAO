import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='SMD',
                    help="dataset from ['ASD', 'MSL'，'SMAP','SMD','SWaT','PSM']")
parser.add_argument('--model',
                    metavar='-m',
                    type=str,
                    required=False,
                    default='HAO',
                    help="model name")
parser.add_argument('--windowsize',
                    metavar='-m',
                    type=int,
                    required=False,
                    default=5,
                    help="windows size")
parser.add_argument('--epoch',
                    metavar='-m',
                    type=int,
                    required=False,
                    default=1,
                    help="train epoch")
parser.add_argument('--space',
                    metavar='-m',
                    type=str,
                    required=False,
                    default="Euclidean",
                    help="space from [Euclidean, Hyperboloid, PoincareBall]")
parser.add_argument('--gpu',
                    metavar='-m',
                    type=int,
                    required=False,
                    default=0,
                    help="gpu index")
parser.add_argument('--test',
                    action='store_true',
                    help="test the model")
parser.add_argument('--retrain',
                    action='store_true',
                    help="retrain the model")
parser.add_argument('--less',
                    action='store_true',
                    help="train using less data")
args = parser.parse_args()
