import argparse

parser = argparse.ArgumentParser(description='video_classifier')
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--rgb-list', default='list/ucf-i3d-train-hand.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default='list/ucf-i3d-test.list', help='list of test rgb features ')
parser.add_argument('--gt', default='list/gt-ucf.npy', help='file of ground truth ')
parser.add_argument('--batch-size', type=int, default=32, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=0, help='number of workers in dataloader')
parser.add_argument('--max-epoch', type=int, default=10, help='maximum iteration to train (default: 15000)')
parser.add_argument('--pre-rgb-list', default='list/pre-ucf-i3d-train-32.list', help='list of rgb preformer features ')
parser.add_argument('--save-mil-name',default='bi-lstm-mil-max',help='save model name (mil model)')
parser.add_argument('--save-videoclassifier-name',default='VideoClasifier-iter-max',help='save model name (video classifier model)')
parser.add_argument('--T',default=32,type=int,help='parameter T')
parser.add_argument('--seed',default=1111,type=int,help='random seed')
parser.add_argument('--test-split-size',default=32,type=int)
parser.add_argument('--da',default=64,type=int)
parser.add_argument('--r',default=3,type=int)