from model_utils import *
import configs
from modules import *


class PhonemeLM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedder = nn.Embedding(
            num_embeddings=configs.phoneme_vocab_size, embedding_dim=configs.phoneme_embedding_dim,
            padding_idx=configs.padding_phoneme_id
        )

        self.encoder = Encoder(input_size=configs.phoneme_embedding_dim, layer_num=2)
        self.classifier = nn.Sequential(
            nn.Linear(configs.rnn_hidden_size, configs.rnn_hidden_size),
            nn.SELU(),
            nn.Dropout(configs.dropout_prob),
            nn.Linear(configs.rnn_hidden_size, configs.phoneme_vocab_size),
        )

        self.init_weights()

    def init_weights(self):
        self.apply(init_weights)

    def forward(
            self,
            # [max_seq_len, batch_size]
            phoneme_seq_batch,
            # [batch_size]
            phoneme_seq_len_batch
    ):
        # print(phoneme_seq_batch.shape)

        # [max_seq_len, batch_size, embedding_dim]
        batch = self.embedder(phoneme_seq_batch)
        # print(batch.shape)

        # print(phoneme_seq_len_batch.shape)
        # print(phoneme_seq_batch.shape)

        # [max_seq_len, batch_size(decreasing), hidden_size
        batch = self.encoder(
            rnn_utils.pack_padded_sequence(batch, phoneme_seq_len_batch)
        )

        # print(len(batch.batch_sizes))

        batch, _ = rnn_utils.pad_packed_sequence(batch)

        # print(batch.shape)

        # [max_seq_len, batch_size, vocab_size]
        scores_seq_batch = self.classifier(batch)

        if self.training:
            return scores_seq_batch
        else:
            # print(scores_seq_batch.shape)
            # [max_seq_len, batch_size, vocab_size]
            # log_probs_seq_batch = F.log_softmax(scores_seq_batch, dim=-1)
            # [batch_size, max_seq_len - 1, vocab_size]
            scores_seq_batch = scores_seq_batch[:-1].contiguous().transpose(0, 1)
            # [batch_size, max_seq_len - 1]
            target_phoneme_seq_batch = phoneme_seq_batch[1:].contiguous().transpose(0, 1)

            # print(log_probs_seq_batch.shape)
            # print(phoneme_seq_batch.shape)
            # print(target_phoneme_seq_batch.shape)

            nll_batch = []

            for scores_seq, target_phoneme_seq in zip(scores_seq_batch, target_phoneme_seq_batch):
                # print(log_probs_seq.shape)
                # print(target_phoneme_seq.shape)

                nll_batch.append(
                    F.cross_entropy(
                        scores_seq, target_phoneme_seq, ignore_index=configs.padding_phoneme_id
                    ).detach().cpu().item()
                )

            return nll_batch



