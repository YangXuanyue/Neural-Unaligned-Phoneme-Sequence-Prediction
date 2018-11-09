import configs
from model_utils import *
from modules import *
import data_utils

"""
BASELINE MODEL 
The model is pretty straight forward and uses BiLSTMs.

3 stacked BiLSTM layers, each of 256 units.
This is followed by a 1 Dense layer of 47 (num_classes) units.
Adam optimizer with default learning rate of 1e-3
Decoding is beam search with a beam width of 100.

No dropouts, batch-normalization or other fancy techniques. The Phoneme Error rate (PER) 
with this baseline model is ~10. CTC loss comes close to ~20 per sample.
 
HOW TO IMPROVE THE BASELINE MODEL

Anything from the HW3P1 paper
CNNs before LSTMs as feature extractors
GRUs/BiGRUs, residual architectures
Addition of extra linear layers after LSTMs
More neurons / More layers in LSTMs
Zoneouts, other regularization techniques
Data preprocessing from HW2P1
Different initial hidden state initializations for the LSTMs.
Different optimizers, learning rates, weight initializations, activations etc
Initialize your dense layer’s bias to the log of the phoneme distribution. 
This helps in faster training of your model.
 
Note that there are many different ways to improve the baseline. While making changes to your model, 
it is extremely useful (and important) to keep track of how each change/update to your model performs. 
This is best done by using loggers as discussed in recitation 4. 

To get a good idea of how low your PER should be, feel free to have a look at last year’s leaderboard.

As always, feel free to ask questions.
"""

def get_model():
    if configs.uses_cnn:
        return CnnRnnModel()
    else:
        return RnnModel()


class RnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.classifier = nn.Linear(
            in_features=configs.rnn_hidden_size,
            out_features=configs.class_num
        )

        self.init_weights()

    def init_classifier(self):
        class_log_priors = data_utils.get_class_log_priors()
        self.classifier.bias.data.copy_(torch.from_numpy(class_log_priors))

    def init_weights(self):
        self.apply(init_weights)
        self.init_classifier()

    def forward(
            self,
            # [max_utterance_len, batch_size(decreasing), frame_size]
            utterance_batch
    ):
        # [max_utterance_len, batch_size(decreasing), rnn_hidden_size]
        hidden_states_batch = self.encoder(utterance_batch)
        # [max_utterance_len, batch_size, rnn_hidden_size], [batch_size]
        hidden_states_batch, utterance_len_batch = rnn_utils.pad_packed_sequence(
            hidden_states_batch
        )
        return self.classifier(hidden_states_batch), utterance_len_batch.int()


class CnnRnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            # v [batch_size, frame_size, utterance_len]
            nn.Conv1d(
                in_channels=configs.frame_size,
                out_channels=(configs.rnn_hidden_size >> 2),
                kernel_size=3, # dilation=2,
                padding=0, stride=2, bias=False
            ),
            # [batch_size, 128, 1 + (utterance_len + 2 * 0 - 2 * (5 - 1) - 1) / 2]
            nn.BatchNorm1d(num_features=(configs.rnn_hidden_size >> 2)),
            nn.ELU(),
            nn.Conv1d(
                in_channels=(configs.rnn_hidden_size >> 2),
                out_channels=(configs.rnn_hidden_size >> 1),
                kernel_size=3,
                padding=1, stride=1, bias=False
            ),
            nn.BatchNorm1d(num_features=(configs.rnn_hidden_size >> 1)),
            nn.ELU(),
            # ResidualConnection(
            #     layer=nn.Sequential(
            #         nn.Conv1d(
            #             in_channels=(configs.rnn_hidden_size >> 1),
            #             out_channels=(configs.rnn_hidden_size >> 1),
            #             kernel_size=3,
            #             padding=1, stride=1, bias=False
            #         ),
            #         nn.BatchNorm1d(num_features=(configs.rnn_hidden_size >> 1)),
            #         nn.ELU()
            #     )
            # )
        )
        self.encoder = ResidualEncoder(input_size=(configs.rnn_hidden_size >> 1)) if configs.uses_residual_rnns \
            else Encoder(input_size=(configs.rnn_hidden_size >> 1))
        # self.highway_feature_extractor = nn.Sequential(
        #     nn.Conv1d(
        #         in_channels=(configs.rnn_hidden_size >> 1),
        #         out_channels=(configs.rnn_hidden_size),
        #         kernel_size=3,
        #         padding=1, stride=1, bias=False
        #     ),
        #     nn.BatchNorm1d(num_features=(configs.rnn_hidden_size)),
        #     nn.ELU(),
        #     nn.Dropout(configs.dropout_prob),
        # )
        # self.highway_encoder = Encoder(input_size=configs.rnn_hidden_size, layer_num=2)
        self.transformer = nn.Sequential(
            nn.Linear(configs.rnn_hidden_size, configs.rnn_hidden_size),
            # nn.BatchNorm1d(num_features=configs.rnn_hidden_size),
            nn.SELU(),
            # v newly added
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.rnn_hidden_size, configs.rnn_hidden_size),
            nn.SELU()
        )
        self.classifier = nn.Linear(
            in_features=configs.rnn_hidden_size,
            out_features=configs.class_num
        )

        self.init_weights()

    def init_classifier(self):
        class_log_priors = data_utils.get_class_log_priors()
        self.classifier.bias.data.copy_(torch.from_numpy(class_log_priors))

    def init_weights(self):
        self.apply(init_weights)
        self.init_classifier()

    def calc_features_seq_len_batch(
            self,
            # [batch_size]
            utterance_len_batch
    ):
        # the len would only be changed by the first conv1d with kernel_size = 5
        # features_seq_len = ((utterance_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        #                  = ((utterance_len + 2 * 0 - 1 * (3 - 1) - 1) / 2) + 1
        #                  = ((utterance_len - 3) / 2) + 1
        return ((utterance_len_batch - 3) // 2) + 1

    def forward(
            self,
            # [batch_size, frame_size], [batch_size]
            utterance_batch, utterance_len_batch
    ):
        # [batch_size, hidden_size / 2, max_features_seq_len]
        features_seq_batch = self.feature_extractor(utterance_batch)
        # [batch_size]
        features_seq_len_batch = self.calc_features_seq_len_batch(utterance_len_batch)

        # features_seq_batch = rnn_utils.pack_padded_sequence(
        #     features_seq_batch, features_seq_len_batch
        # )
        # [max_features_seq_len, batch_size(decreasing), rnn_hidden_size]
        hidden_states_batch = self.encoder(
            rnn_utils.pack_padded_sequence(
                # [max_features_seq_len, batch_size(decreasing), hidden_size / 2
                features_seq_batch.permute(2, 0, 1), features_seq_len_batch
            )
        )

        # highway_hidden_states_batch = self.highway_encoder(
        #     rnn_utils.pack_padded_sequence(
        #         # [max_features_seq_len, batch_size(decreasing), hidden_size / 2
        #         self.highway_feature_extractor(features_seq_batch).permute(2, 0, 1), features_seq_len_batch
        #     )
        # )

        if isinstance(hidden_states_batch, rnn_utils.PackedSequence):
            # [max_features_seq_len, batch_size, rnn_hidden_size], [batch_size]
            hidden_states_batch, _ = rnn_utils.pad_packed_sequence(
                hidden_states_batch
            )

        # if isinstance(highway_hidden_states_batch, rnn_utils.PackedSequence):
        #     # [max_features_seq_len, batch_size, rnn_hidden_size], [batch_size]
        #     highway_hidden_states_batch, _ = rnn_utils.pad_packed_sequence(
        #         highway_hidden_states_batch
        #     )



        return self.classifier(
            self.transformer(
                hidden_states_batch
                # # [max_features_seq_len, batch_size, rnn_hidden_size * 2]
                # torch.cat(
                #     (
                #         # [max_features_seq_len, batch_size, rnn_hidden_size]
                #         hidden_states_batch,
                #         # [max_features_seq_len, batch_size, hidden_size]
                #         highway_hidden_states_batch
                #     ), dim=-1
                # )
            )
        ), features_seq_len_batch.int()
