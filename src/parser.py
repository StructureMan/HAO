import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset',
                    metavar='-d',
                    type=str,
                    required=False,
                    default='SMD',
                    help="dataset from ['SMD', 'SMAP'，'MSL','ASD','SWaT','WADI']")
parser.add_argument('--model',
                    metavar='-m',
                    type=str,
                    required=False,
                    default='GSL_AD',
                    help="model name")
parser.add_argument('--epoch',
                    metavar='-e',
                    type=int,
                    required=False,
                    default=1,
                    help="number of epochs")
parser.add_argument('--window_size',
                    metavar='-w',
                    type=int,
                    required=False,
                    default=5,
                    help="window size")

parser.add_argument('--description',
                    metavar='-d',
                    type=str,
                    required=False,
                    default="Demo",
                    help="experiment description")
parser.add_argument('--is_corrected',
                    metavar='-c',
                    type=int,
                    required=False,
                    default=1,
                    help="whether to use corrected labels")
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
