import configs
import os

from tqdm import tqdm
from model_utils import *
import random
from collections import defaultdict, Counter
from itertools import chain
import time
import json
import Levenshtein
from data import phoneme_list


def load_raw_dataset(name):
    utterances = np.load(f'data/wsj0_{name}.npy', encoding='latin1')
    label_seqs = np.load(f'data/wsj0_{name}_merged_labels.npy') if name != 'test' \
        else np.array([[-1] for _ in range(len(utterances))])
    # leave label 0 to be the <blank>
    label_seqs += 1
    return utterances, label_seqs


names = (('test',), ('train', 'dev'))

raw_datasets = {
    name: load_raw_dataset(name)
    for name in names[configs.training]
}


class Dataset(tud.Dataset):
    def __init__(self, name):
        self.name = name
        self.utterances, self.label_seqs = raw_datasets[self.name]

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        return self.utterances[idx], self.label_seqs[idx].astype(np.int32), idx


class PhonemeDataset(tud.Dataset):
    def __init__(self, name):
        self.name = name
        _, self.label_seqs = raw_datasets[self.name]
        self.label_seqs = [
            np.array(
                [configs.start_phoneme_id] + label_seq.tolist() + [configs.end_phoneme_id]
            )
            for label_seq in self.label_seqs
        ]

    def __len__(self):
        return len(self.label_seqs)

    def __getitem__(self, idx):
        return self.label_seqs[idx]


datasets = {
    name: Dataset(name)
    for name in names[configs.training]
}

phoneme_datasets = {
    name: PhonemeDataset(name)
    for name in names[configs.training]
}


def get_data_size(name):
    return len(datasets[name])


def get_class_log_priors():
    if configs.training and not os.path.exists('class_log_priors.npy'):
        label_cnts = Counter(chain(*datasets['train'].label_seqs))
        class_log_priors = np.zeros(configs.class_num)
        assert configs.blank_label not in label_cnts
        blank_cnt = sum(label_cnts.values())  # / len(label_cnts)
        label_cnts[configs.blank_label] = blank_cnt
        log_tot_cnt = math.log(sum(label_cnts.values()))

        for label, cnt in label_cnts.items():
            class_log_priors[label] = math.log(cnt) - log_tot_cnt

        np.save('class_log_priors.npy', class_log_priors)
    else:
        class_log_priors = np.load('class_log_priors.npy')

    return class_log_priors


def get_phonemes():
    return phoneme_list.PHONEME_MAP


def convert_to_phoneme_str(label_seq):
    if isinstance(label_seq, str):
        return label_seq
    else:
        return ''.join(map(phoneme_list.PHONEME_MAP.__getitem__, label_seq))


def get_levenshtein_distance(label_seq_1, label_seq_2):
    return Levenshtein.distance(
        *map(convert_to_phoneme_str, (label_seq_1, label_seq_2))
    )


def save_submission(best_prediction_seqs):
    with open('submission.csv', 'w') as submission_file:
        print('Id,Predicted', file=submission_file)
        print(
            '\n'.join(
                f'{id_},{phoneme_str}'
                for id_, phoneme_str in enumerate(map(convert_to_phoneme_str, best_prediction_seqs))
            ),
            file=submission_file
        )


# def pad_utterances(utterances):
#     max_utterance_len = max(len(utterance) for utterance in utterances)
#
#     return [
#         np.concatenate(
#             (utterance, np.zeros((max_utterance_len - len(utterance), configs.frame_size))),
#             axis=0
#         ) if len(utterance) < max_utterance_len else utterance
#         for utterance in utterances
#     ]
#
#
# def transpose_utterances(utterances):
#     if isinstance(utterances, list):
#         return [
#             utterance.T
#             for utterance in utterances
#         ]
#     elif isinstance(utterances, torch.Tensor):
#         return utterances.transpose(dim0=1, dim1=2)
#     else:
#         return utterances


def collate(batch):
    if configs.uses_cnn:
        batch = sorted(batch, key=(lambda u_ls_i: len(u_ls_i[0])), reverse=True)
        utterance_batch, label_seq_batch, idx_batch = zip(*batch)
        utterance_len_batch = torch.LongTensor(
            [len(utterance) for utterance in utterance_batch]
        )
        # utterance_batch is now of size [batch_size, utterance_len(decreasing), frame_size]
        # pad it to [batch_size, max_utterance_len, frame_size]
        # and transpose it to [batch_size, frame_size, max_utterance_len]
        max_utterance_len = len(utterance_batch[0])
        utterance_batch = torch.cuda.FloatTensor(
            [
                np.concatenate(
                    (utterance, np.zeros((max_utterance_len - len(utterance), configs.frame_size))),
                    axis=0
                ).T if len(utterance) < max_utterance_len else utterance.T
                for utterance in utterance_batch
            ]
        )
        if configs.training:
            label_seq_len_batch = torch.IntTensor(
                [len(label_seq) for label_seq in label_seq_batch]
            )
            flat_label_seq_batch = torch.from_numpy(
                np.concatenate(label_seq_batch)
            )

            return (
                # [batch_size, max_utterance_len, frame_size]
                utterance_batch,
                # [batch_size]
                utterance_len_batch,
                # [sum(i, len(label_seq_batch[i]))]
                flat_label_seq_batch,
                # [batch_size]
                label_seq_len_batch
            )
        else:
            return utterance_batch, utterance_len_batch, idx_batch

    else:
        batch = sorted(batch, key=(lambda u_ls_i: len(u_ls_i[0])), reverse=True)
        utterance_batch, label_seq_batch, idx_batch = zip(*batch)
        # utterance_len_batch = torch.LongTensor(
        #     [len(utterance) for utterance in utterance_batch]
        # )
        utterance_batch = rnn_utils.pack_sequence(
            [
                torch.from_numpy(utterance).cuda()
                for utterance in utterance_batch
            ]
        )

        if configs.training:
            label_seq_len_batch = torch.IntTensor(
                [len(label_seq) for label_seq in label_seq_batch]
            )
            flat_label_seq_batch = torch.from_numpy(
                np.concatenate(label_seq_batch)
            )

            return (
                # [max_utterance_len, batch_size(decreasing), frame_size]
                utterance_batch,
                # # [batch_size]
                # utterance_len_batch,
                # [sum(i, len(label_seq_batch[i]))]
                flat_label_seq_batch,
                # [batch_size]
                label_seq_len_batch
            )
        else:
            return utterance_batch, idx_batch


data_loaders = {
    name: tud.DataLoader(
        dataset=datasets[name],
        batch_size=configs.batch_size,
        shuffle=(name == 'train'),
        # pin_memory=True,
        collate_fn=collate,
        num_workers=0
    )
    for name in names[configs.training]
}


def collate_phoneme_seq_batch(batch):
    phoneme_seq_batch = sorted(batch, key=len, reverse=True)
    phoneme_seq_len_batch = torch.LongTensor(
        [len(phoneme_seq) for phoneme_seq in phoneme_seq_batch]
    )
    max_phoneme_seq_len = len(phoneme_seq_batch[0])
    # [max_utterance_len, batch_size]
    phoneme_seq_batch = torch.cuda.LongTensor(
        [
            np.concatenate(
                (phoneme_seq, np.zeros((max_phoneme_seq_len - len(phoneme_seq)))),
                axis=0
            ) if len(phoneme_seq) < max_phoneme_seq_len else phoneme_seq
            for phoneme_seq in phoneme_seq_batch
        ]
    ).transpose(0, 1)

    return phoneme_seq_batch, phoneme_seq_len_batch


phoneme_data_loaders = {
    name: tud.DataLoader(
        dataset=phoneme_datasets[name],
        batch_size=configs.batch_size,
        shuffle=(name == 'train'),
        # pin_memory=True,
        collate_fn=collate_phoneme_seq_batch,
        num_workers=0
    )
    for name in names[configs.training]
}


def gen_batches(name):
    input_num = 0

    for batch in data_loaders[name]:
        # label_seq_len_batch: [batch_size]
        input_num += len(batch[-1])
        pct = input_num * 100. / get_data_size(name)
        # print(pct)
        yield pct, batch

def gen_phoneme_seq_batches(name):
    input_num = 0

    for batch in phoneme_data_loaders[name]:
        # label_seq_len_batch: [batch_size]
        input_num += len(batch[-1])
        pct = input_num * 100. / get_data_size(name)
        # print(pct)
        yield pct, batch
