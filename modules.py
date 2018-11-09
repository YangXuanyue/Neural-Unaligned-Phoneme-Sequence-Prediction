from model_utils import *
from collections import Counter, defaultdict
import data_utils
from ctcdecode import CTCBeamDecoder
import configs


# zoneout
# https://github.com/eladhoffer/seq2seq.pytorch/blob/master/seq2seq/models/modules/recurrent.py

class ResidualConnection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, input):
        return self.layer(input) + input


class Encoder(nn.Module):
    def __init__(self, input_size=configs.frame_size, layer_num=configs.rnn_layer_num):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = (configs.rnn_hidden_size // 2 if configs.uses_bi_rnn else configs.rnn_hidden_size)
        self.layer_num = layer_num
        self.direction_num = 2 if configs.uses_bi_rnn else 1
        self.rnn_type = nn.LSTM if configs.rnn_type == 'lstm' else nn.GRU
        self.rnn = self.rnn_type(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.layer_num,
            bidirectional=configs.uses_bi_rnn,
            # dropout=configs.dropout_prob
        )
        self.cached_rnn_weights = {}
        self.state_shape = [self.layer_num * self.direction_num, 1, self.hidden_size]
        self.batch_dim = 1

        if self.rnn_type is nn.LSTM:
            self.initial_hidden_state = nn.Parameter(torch.randn(*self.state_shape))
            self.initial_cell_state = nn.Parameter(torch.randn(*self.state_shape))
        elif self.rnn_type is nn.GRU:
            self.initial_hidden_state = nn.Parameter(torch.randn(*self.state_shape))

        if configs.uses_weight_dropped_rnn:
            from weight_drop import WeightDrop

            weight_names = [name for name, param in self.rnn.named_parameters() if 'weight' in name]
            self.rnn = WeightDrop(module=self.rnn, weights=weight_names, dropout=configs.rnn_weights_dropout_prob)

    def get_initial_state(self, batch_size):
        # self.state_shape[self.batch_dim] = batch_size

        if self.rnn_type is nn.LSTM:
            return (
                self.initial_hidden_state.repeat(1, batch_size, 1),
                self.initial_cell_state.repeat(1, batch_size, 1)
            )
        elif self.rnn_type is nn.GRU:
            return self.initial_hidden_state.repeat(1, batch_size, 1)

    def forward(self, utterance_batch):
        if isinstance(utterance_batch, rnn_utils.PackedSequence):
            batch_size = utterance_batch.batch_sizes[0]
        else:
            _, batch_size, _ = utterance_batch.shape

        # [max_utterance_len, batch_size(decreasing), rnn_hidden_size]
        hidden_states_batch, *_ = self.rnn(
            utterance_batch, self.get_initial_state(batch_size)
        )

        return hidden_states_batch


class RnnModule(nn.Module):
    def __init__(self, input_size=configs.frame_size, dropout_prob=None, residual=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = (configs.rnn_hidden_size // 2 if configs.uses_bi_rnn else configs.rnn_hidden_size)
        # self.layer_num = configs.rnn_layer_num
        self.direction_num = 2 if configs.uses_bi_rnn else 1
        self.rnn_type = nn.LSTM if configs.rnn_type == 'lstm' else nn.GRU
        self.rnn = self.rnn_type(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            # num_layers=self.layer_num,
            bidirectional=configs.uses_bi_rnn,
            # dropout=configs.dropout_prob,
            bias=False
        )
        # self.selu = nn.SELU()
        self.dropout_prob = dropout_prob
        self.residual = residual

        # if self.residual:
        #     self.batch_norm = nn.BatchNorm1d(configs.rnn_hidden_size)

        # if self.dropout_prob is not None:
        #     self.dropout = nn.Dropout(self.dropout_prob)

        self.state_shape = [self.direction_num, 1, self.hidden_size]
        # self.batch_dim = 1

        if self.rnn_type is nn.LSTM:
            self.initial_hidden_state = nn.Parameter(torch.randn(*self.state_shape))
            self.initial_cell_state = nn.Parameter(torch.randn(*self.state_shape))
        elif self.rnn_type is nn.GRU:
            self.initial_hidden_state = nn.Parameter(torch.randn(*self.state_shape))

    def get_initial_state(self, batch_size):
        # self.state_shape[self.batch_dim] = batch_size

        if self.rnn_type is nn.LSTM:
            return (
                self.initial_hidden_state.repeat(1, batch_size, 1),
                self.initial_cell_state.repeat(1, batch_size, 1)
            )
        elif self.rnn_type is nn.GRU:
            return self.initial_hidden_state.repeat(1, batch_size, 1)

    def forward(self, input_batch, len_batch):
        if isinstance(input_batch, rnn_utils.PackedSequence):
            batch_size = input_batch.batch_sizes[0]
            packed_input_batch = input_batch
        else:
            # print(type(input_batch))
            # print(input_batch)
            _, batch_size, _ = input_batch.shape
            packed_input_batch = rnn_utils.pack_padded_sequence(input_batch, len_batch)

        # [max_utterance_len, batch_size(decreasing), rnn_hidden_size]
        hidden_states_batch, *_ = self.rnn(
            packed_input_batch,
            self.get_initial_state(batch_size)
        )

        # [max_utterance_len, batch_size, rnn_hidden_size]
        hidden_states_batch, len_batch = rnn_utils.pad_packed_sequence(hidden_states_batch)

        # hidden_states_batch = self.batch_norm(hidden_states_batch)
        # hidden_states_batch = self.selu(hidden_states_batch)

        if self.residual:
            hidden_states_batch = hidden_states_batch + input_batch

            # hidden_states_batch = self.batch_norm(
            #     (hidden_states_batch + input_batch).view(-1, configs.rnn_hidden_size)
            # ).view(*hidden_states_batch.shape)

        # if self.dropout_prob is not None:
        #     hidden_states_batch = self.dropout(hidden_states_batch)

        return hidden_states_batch, len_batch


class ResidualEncoder(nn.Module):
    def __init__(self, input_size=configs.frame_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = (configs.rnn_hidden_size // 2 if configs.uses_bi_rnn else configs.rnn_hidden_size)
        self.layer_num = configs.rnn_layer_num
        self.direction_num = 2 if configs.uses_bi_rnn else 1
        self.rnn_type = nn.LSTM if configs.rnn_type == 'lstm' else nn.GRU
        self.state_shape = [self.direction_num, 1, self.hidden_size]
        self.rnns = nn.ModuleList(
            [
                RnnModule(
                    input_size=(configs.rnn_hidden_size if i > 0 else self.input_size),
                    # dropout_prob=(configs.dropout_prob if i < self.layer_num - 1 else None),
                    residual=(configs.uses_residual_rnns and i > 0)
                )
                for i in range(self.layer_num)
            ]
        )

    def forward(self, utterance_batch):
        # print(f'utterance_batch: {type(utterance_batch)}')
        states_batch = utterance_batch
        len_batch = None

        for i in range(self.layer_num):
            # print(i)
            # print(f'states_batch: {type(states_batch)}')
            # [max_utterance_len, batch_size(decreasing), rnn_hidden_size]
            states_batch, len_batch = self.rnns[i](states_batch, len_batch)

        return states_batch


class BatchNormEncoder(nn.Module):
    def __init__(self, input_size=configs.frame_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = (configs.rnn_hidden_size // 2 if configs.uses_bi_rnn else configs.rnn_hidden_size)
        self.layer_num = configs.rnn_layer_num
        self.direction_num = 2 if configs.uses_bi_rnn else 1
        self.rnn_type = nn.LSTM if configs.rnn_type == 'lstm' else nn.GRU
        self.state_shape = [self.direction_num, 1, self.hidden_size]
        self.rnns = nn.ModuleList(
            [
                RnnModule(
                    input_size=(configs.rnn_hidden_size if i > 0 else self.input_size),
                    dropout_prob=(configs.l2_weight_decay if i < self.layer_num - 1 else None),
                    residual=(configs.uses_residual_rnns and i > 0)
                )
                for i in range(self.layer_num)
            ]
        )
        # self.batch_norms = nn.ModuleList()
        # self.selus = nn.ModuleList()

        # if self.rnn_type is nn.LSTM:
        #     self.initial_hidden_states = nn.ModuleList()
        #     self.initial_cell_states = nn.ModuleList()
        # elif self.rnn_type is nn.GRU:
        #     self.initial_hidden_states = nn.ModuleList()
        #
        #
        # for i in range(configs.rnn_layer_num):
        #     if self.rnn_type is nn.LSTM:
        #         self.initial_hidden_states.append(nn.Parameter(torch.randn(*self.state_shape)))
        #         self.initial_cell_states.append(nn.Parameter(torch.randn(*self.state_shape)))
        #     elif self.rnn_type is nn.GRU:
        #         self.initial_hidden_states.append(nn.Parameter(torch.randn(*self.state_shape)))
        #
        #     self.rnns.append(
        #         self.rnn_type(
        #             input_size=self.input_size if i > 0 else self.input_size,
        #             hidden_size=self.hidden_size,
        #             # num_layers=self.layer_num,
        #             bidirectional=configs.uses_bi_rnn
        #         )
        #     )
        #     self.batch_norms.append(nn.BatchNorm1d(self.hidden_size))
        #     self.selus.append(nn.SELU())
        # #
        #
        # self.rnn = self.rnn_type(
        #     input_size=self.input_size,
        #     hidden_size=self.hidden_size,
        #     num_layers=self.layer_num,
        #     bidirectional=configs.uses_bi_rnn
        # )
        # self.batch_dim = 1

    # def get_initial_state_batch(self, batch_size, layer_idx):
    #     if self.rnn_type is nn.LSTM:
    #         return (
    #             self.initial_hidden_states[layer_idx].repeat(1, batch_size, 1),
    #             self.initial_cell_states[layer_idx].repeat(1, batch_size, 1)
    #         )
    #     elif self.rnn_type is nn.GRU:
    #         return self.initial_hidden_states[layer_idx].repeat(1, batch_size, 1)

    def forward(self, utterance_batch):
        # if isinstance(utterance_batch, rnn_utils.PackedSequence):
        #     batch_size = utterance_batch.batch_sizes[0]
        # else:
        #     _, batch_size, _ = utterance_batch.shape

        states_batch = utterance_batch
        len_batch = None
        # states_batches = []

        for i in range(self.layer_num):
            # [max_utterance_len, batch_size(decreasing), rnn_hidden_size]
            states_batch, len_batch = self.rnns[i](states_batch, len_batch)
            # states_batch = self.batch_norms[i](states_batch)
            # states_batch = self.selus[i](states_batch)

            # if i > 0 and configs.uses_residual_rnns:
            #     states_batch += states_batches[-1]

            # states_batches.append(states_batch)

            # if i < self.layer_num - 1:
            #     states_batch = rnn_utils.pack_padded_sequence(states_batch, len_batch)

        return states_batch


class Decoder:
    def __init__(self, best=True):
        self.ctc_beam_decoder = CTCBeamDecoder(
            labels=data_utils.get_phonemes(),
            blank_id=configs.blank_label,
            beam_width=configs.beam_width,
        )
        self.best = best

        if configs.uses_lm:
            from phoneme_lm import PhonemeLM
            self.lm = PhonemeLM()
            ckpt = torch.load('lm_ckpts/1101-223757.1.ckpt')
            self.lm.load_state_dict(ckpt['model'])
            print('loaded phoneme language model')
            self.lm.eval()

    def __call__(self, output_batch, utterance_len_batch):
        return self.forward(output_batch, utterance_len_batch)

    def forward(
            self,
            # [max_utterance_len, batch_size, class_num]
            output_batch,
            # [batch_size]
            utterance_len_batch
    ):
        # [batch_size, max_utterance_len, class_num]
        output_batch = output_batch.transpose(0, 1)
        batch_size, *_ = output_batch.shape
        # [batch_size, max_utterance_len, class_num]
        prob_seq_batch = F.softmax(output_batch, dim=-1)
        # [batch_size, beam_width, max_utterance_len], [batch_size, beam_width], _, [batch_size, beam_width]
        prediction_seq_beam_batch, score_beam_batch, _, prediction_seq_len_beam_batch = self.ctc_beam_decoder.decode(
            prob_seq_batch, utterance_len_batch
        )

        if configs.uses_lm:
            best_prediction_seq_batch = []

            with torch.no_grad():
                for prediction_seq_beam, score_beam, prediction_seq_len_beam in zip(
                        prediction_seq_beam_batch, score_beam_batch, prediction_seq_len_beam_batch
                ):
                    # print(prediction_seq_beam.shape)

                    beam_width = len(prediction_seq_beam)

                    prediction_seq_beam = [
                        [configs.start_phoneme_id]
                        + prediction_seq_beam[i][:prediction_seq_len_beam[i]].detach().cpu().numpy().tolist()
                        + [configs.end_phoneme_id]
                        for i in range(beam_width)
                    ]

                    prediction_seq_len_beam = [
                        prediction_seq_len.detach().cpu().item() + 2
                        for prediction_seq_len in prediction_seq_len_beam
                    ]

                    max_prediction_seq_len = max(prediction_seq_len_beam)


                    for i in range(len(prediction_seq_beam)):
                        prediction_seq_beam[i] += \
                            [configs.end_phoneme_id] * (max_prediction_seq_len - prediction_seq_len_beam[i])

                    something = [
                        (prediction_seq_beam[i], score_beam[i], prediction_seq_len_beam[i])
                        for i in range(len(prediction_seq_beam))
                    ]

                    something = sorted(something, key=(lambda x: -x[-1]))

                    prediction_seq_beam, score_beam, prediction_seq_len_beam = zip(*something)

                    prediction_seq_beam = torch.LongTensor(prediction_seq_beam)

                    # print(prediction_seq_beam.shape)


                    nll_beam = self.lm(
                        # [max_seq_len, beam_width]
                        prediction_seq_beam.transpose(0, 1),
                        # [beam_width]
                        torch.cuda.LongTensor(prediction_seq_len_beam)
                    )

                    # for nll, score in zip(nll_beam, score_beam):
                    #     print((nll, score))

                    avg_score = sum(score_beam) / len(score_beam)
                    avg_nll = sum(nll_beam) / len(nll_beam)
                    ll_coef = abs(avg_score / avg_nll)
                    ll_beam = -abs(torch.FloatTensor(nll_beam)) * ll_coef
                    score_beam = np.array(score_beam) + ll_beam
                    best_seq_idx = np.argmax(score_beam)

                    best_prediction_seq_batch.append(
                        prediction_seq_beam[best_seq_idx][1:(prediction_seq_len_beam[best_seq_idx] - 1)]
                            .detach().cpu().numpy().tolist()
                    )

                return best_prediction_seq_batch



        if self.best:
            # [prediction_seq_len] * batch
            return [
                # [prediction_seq_len]
                prediction_seq_beam[0][:prediction_seq_len_beam[0]]
                for prediction_seq_beam, prediction_seq_len_beam in zip(
                    prediction_seq_beam_batch, prediction_seq_len_beam_batch
                )
            ]
        else:
            # [batch_size, beam_width, max_utterance_len], [batch_size, beam_width]
            return prediction_seq_beam_batch, prediction_seq_len_beam_batch


class ResidualBlock(nn.Module):
    """
    All conv2d layers have bias=False
    Each residual block consists of cov3x3, bn, ELU, conv3x3, bn, ELU (of course, with the residual connection)
    All conv3x3 have padding=1, stride=1
    """

    def __init__(self, channel_num):
        super().__init__()

        self.main_branch = nn.Sequential(
            # such a conv3x3 won't change the size of the input
            nn.Conv2d(in_channels=channel_num, out_channels=channel_num,
                      kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=channel_num),
            nn.ELU(),
            nn.Conv2d(in_channels=channel_num, out_channels=channel_num,
                      kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=channel_num),
        )
        self.activation = nn.ELU()

    def forward(self, input_batch):
        return self.activation(self.main_branch(input_batch)) + input_batch


class ResidualLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, input):
        return self.layer(input) + input


class BasicModule(nn.Module):
    """
    All conv2d layers have bias=False
    conv5x5, ELU, residual block
    All conv5x5 have padding=0, stride=2
    """

    def __init__(self, in_channel_num, out_channel_num, kernel_size=5):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channel_num, out_channel_num, kernel_size=kernel_size, padding=0, stride=2, bias=False),
            nn.ELU(),
            ResidualBlock(out_channel_num)
        )

    def forward(self, input_batch):
        return self.layers(input_batch)


class Reshaper(nn.Module):
    def __init__(self, output_shape):
        super().__init__()

        self.output_shape = output_shape

    def forward(self, input: torch.Tensor):
        return input.view(*self.output_shape)


class Scaler(nn.Module):
    def __init__(self, alpha=16.):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha).cuda())

    def forward(self, input):
        return self.alpha * input


class Normalizer(nn.Module):
    def __init__(self, target_norm=1.):
        super().__init__()
        self.target_norm = target_norm

    def forward(self, input: torch.Tensor):
        return input * self.target_norm / input.norm(p=2, dim=1, keepdim=True)


class MeanStdExtractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_maps_batch):
        # [batch_size, feature_num, width * height]
        feature_maps_batch = feature_maps_batch.view(*feature_maps_batch.shape[:2], -1)
        # [batch_size, feature_num]
        feature_means_batch = feature_maps_batch.mean(dim=-1)
        # [batch_size, feature_num]
        feature_stds_batch = feature_maps_batch.std(dim=-1)
        # [batch_size, feature_num * 2]
        return torch.cat((feature_means_batch, feature_stds_batch), dim=-1)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=-.6):
        super().__init__()
        self.margin = margin

    def forward(self, score_batch, label_batch):
        batch_size, = score_batch.shape
        similar_loss_batch = (1. - score_batch[label_batch == 1]) ** 2.
        dissimilar_loss_batch = (score_batch[label_batch == 0] - self.margin).clamp(min=0.) ** 2.

        return (similar_loss_batch.sum() + dissimilar_loss_batch.sum()) / batch_size


class CenterLoss(nn.Module):
    def __init__(self, class_num, feature_num, alpha=.5):
        super(CenterLoss, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num
        self.class_centers = nn.Parameter(torch.randn(self.class_num, self.feature_num).cuda())
        # self.alpha = alpha

    def forward(self, embedding_batch, label_batch):
        # [batch_size, feature_num]
        label_center_batch = self.class_centers[label_batch]
        # [batch_size, feature_num]
        diff_batch = embedding_batch - label_center_batch
        # [batch_size, feature_num]
        loss = (diff_batch ** 2.).sum(dim=1).mean()

        # diff_batch = diff_batch.detach()
        # label_batch = label_batch.detach()
        # center_changes = defaultdict(lambda: torch.zeros(self.feature_num).cuda())
        # class_cnts = defaultdict(float)
        #
        # for diff, label in zip(diff_batch, label_batch):
        #     center_changes[label] += diff
        #     class_cnts[label] += 1.
        #
        # for class_ in center_changes:
        #     self.class_centers[class_] += self.alpha * center_changes[class_] / (1. + class_cnts[class_])

        return loss


class Loss(nn.Module):
    def __init__(self, class_num, feature_num, lambda_=.01):
        super().__init__()

        self.x_ent_loss = nn.CrossEntropyLoss()
        self.lambda_ = lambda_
        self.center_loss = CenterLoss(class_num, feature_num)

    def forward(self, embedding_batch, output_batch, label_batch):
        x_ent_loss = self.x_ent_loss(output_batch, label_batch)
        center_loss = self.center_loss(embedding_batch, label_batch)

        # print(f'x_ent_loss = {x_ent_loss.detach().cpu().numpy().item()}')
        # print(f'lambda_ * center_loss = {(self.lambda_ * center_loss).detach().cpu().numpy().item()}')

        return x_ent_loss + self.lambda_ * center_loss

# char_embed_size = 4
# output_size = 128
# max_seq_length = 128
#
#
# class ParallelModule(nn.Module):
#
#     def __init__(self, branches):
#         super().__init__()
#
#         self.branches = nn.ModuleList(branches)
#
#     def forward(self, emb):
#         # old way
#         tmp = [cnn(emb).squeeze() for cnn in self.cnns]
#         seq_representation = torch.cat(tmp, dim=1)
#
#         # new way
#         stream_tmp = []
#         streams = [(idx, torch.cuda.Stream()) for idx, cnn in enumerate(self.cnns)]
#         for idx, s in streams:
#             with torch.cuda.stream(s):
#                 cnn = self.cnns[idx]  # <--- how to ensure idx is in sync with the idx in for loop?
#                 stream_tmp.append((idx, cnn(emb).squeeze()))
#         torch.cuda.synchronize()  # added synchronize
#         stream_tmp = [t for idx, t in sorted(stream_tmp)]
#         seq_representation_stream = torch.cat(stream_tmp, dim=1)
#
#         # comparing the two
#         diff = abs(seq_representation_stream - seq_representation).sum().data[0]
#         print(diff)
#         assert diff == 0.0
#         return seq_representation
