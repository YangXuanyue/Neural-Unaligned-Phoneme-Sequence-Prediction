import datetime
import argparse as ap
from model_utils import *
import time
import random
import os

seed = int(time.time())
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)



arg_parser = ap.ArgumentParser()
arg_parser.add_argument('-m', '--mode', default='train')
arg_parser.add_argument('-c', '--ckpt', default=None)
# arg_parser.add_argument('-d', '--device', type=int, default=0)
arg_parser.add_argument('-b', action='store_true', default=False)
arg_parser.add_argument('-l', action='store_true', default=False)

args = arg_parser.parse_args()

timestamp = datetime.datetime.now().strftime('%m%d-%H%M%S')

class Dir:
    def __init__(self, name):
        self.name = name

        if not os.path.exists(name):
            os.mkdir(name)

    def __str__(self):
        return self.name


logs_dir = Dir('logs')
ckpts_dir = Dir('ckpts')
lm_ckpts_dir = Dir('lm_ckpts')

training = args.mode == 'train'
# ckpt.best is trained with data_part_num = 3
ckpt_id = args.ckpt
# device_id = args.device
loads_best_ckpt = args.b
loads_ckpt = args.l

transposes_utterance = False
adds_channel_dim = True

# max_frame_num = 14_000
frame_size = 40
# channel_num = 1
blank_label = 0
phoneme_num = 46
phoneme_embedding_dim = 256
start_phoneme_id = phoneme_num
end_phoneme_id = phoneme_num + 1
padding_phoneme_id = phoneme_num + 2
# [...] + '<s>' + '</s>' + '<pad>'
phoneme_vocab_size = phoneme_num + 3
# ' ' + [...]
class_num = 1 + phoneme_num
lr = 1e-7
lm_lr = 5e-5
sets_new_lr = True
batch_size = 64
uses_bi_rnn = True
rnn_type = 'lstm'
# rnn_type = 'gru'
rnn_hidden_size = 512
rnn_layer_num = 4
min_logit = -1e3
beam_width = 100
dropout_prob = .2
momentum = .9
l2_weight_decay = 5e-3
feature_num = 512

epoch_num = 240

uses_center_loss = False
uses_pairwise_sim_loss = False

normalizes_outputs = True
uses_new_classifier = False
uses_new_transformer = False
uses_new_encoder = False

uses_cnn = True

uses_batch_norm_encoder = False
uses_residual_rnns = False

loads_optimizer_state = not (sets_new_lr or uses_new_classifier or uses_new_encoder)

lr_scheduler_factor = .8
lr_scheduler_patience = 0

uses_weight_dropped_rnn = True

rnn_weights_dropout_prob = .2

tests_ensemble = False

uses_lm = not training and False