import configs
from model import get_model
from modules import Decoder
from model_utils import *
from warpctc_pytorch import CTCLoss
import os
from log import log
import time
import data_utils
import re
from itertools import chain


class Runner:
    def __init__(self):
        self.model = get_model().cuda()
        self.ctc_loss = CTCLoss(size_average=True)
        self.decoder = Decoder()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=configs.lr, weight_decay=configs.l2_weight_decay)
        self.optimizer = optim.ASGD(self.model.parameters(), lr=configs.lr, weight_decay=configs.l2_weight_decay)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',
                                                           patience=configs.lr_scheduler_patience,
                                                           factor=configs.lr_scheduler_factor, verbose=True)
        self.epoch_idx = 0
        self.min_avg_dist = 1000.

    @staticmethod
    def calc_avg_dist(
            # [batch_size, max_utterance_len]
            best_prediction_seq_batch,
            # [# [sum(i, len(label_seq_batch[i]))], [batch_size]
            flat_label_seq_batch, label_seq_len_batch

    ):
        batch_size = len(best_prediction_seq_batch)
        dist_sum = 0.
        start_idx = 0

        for best_prediction_seq, label_seq_len in zip(
                best_prediction_seq_batch, label_seq_len_batch
        ):
            dist_sum += data_utils.get_levenshtein_distance(
                best_prediction_seq, flat_label_seq_batch[start_idx:(start_idx + label_seq_len)]
            )
            start_idx += label_seq_len

        return dist_sum / batch_size

    def train(self):
        if configs.ckpt_id or configs.loads_ckpt:
            self.load_ckpt()

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        start_epoch_idx = self.epoch_idx

        for epoch_idx in range(start_epoch_idx, configs.epoch_num):
            self.epoch_idx = epoch_idx

            log(f'starting epoch {epoch_idx}')
            log('training')

            self.model.train()
            avg_epoch_loss = 0.
            batch_num = 0
            next_logging_pct = .5

            start_time = time.time()

            for pct, batch in data_utils.gen_batches('train'):
                batch_num += 1
                self.optimizer.zero_grad()

                if configs.uses_cnn:
                    utterance_batch, utterance_len_batch, flat_label_seq_batch, label_seq_len_batch = batch
                    # [max_utterance_len, batch_size, class_num], [batch_size]
                    output_batch, features_seq_len_batch = self.model.forward(utterance_batch, utterance_len_batch)
                    loss = self.ctc_loss(
                        output_batch,
                        flat_label_seq_batch,
                        features_seq_len_batch,
                        label_seq_len_batch
                    )
                else:
                    utterance_batch, flat_label_seq_batch, label_seq_len_batch = batch
                    # [max_utterance_len, batch_size, class_num], [batch_size]
                    output_batch, utterance_len_batch = self.model.forward(utterance_batch)
                    loss = self.ctc_loss(
                        output_batch,
                        flat_label_seq_batch,
                        utterance_len_batch,
                        label_seq_len_batch
                    )

                # print(torch.argmax(output_batch[:50, :5, :], dim=-1))

                loss.backward()
                self.optimizer.step()
                avg_epoch_loss += loss.detach().cpu().item()

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, avg_train_loss: {avg_epoch_loss / batch_num}, '
                        f'time: {time.time() - start_time}'
                    )
                    next_logging_pct += 10.

            avg_epoch_loss /= batch_num

            log(
                f'avg_train_loss: {avg_epoch_loss}\n'
                f'avg_train_time: {time.time() - start_time}'
            )

            with torch.no_grad():
                log('validating')

                self.model.eval()
                batch_num = 0
                avg_dist = 0
                next_logging_pct = 10.

                start_time = time.time()

                for pct, batch in data_utils.gen_batches('dev'):
                    batch_num += 1
                    self.optimizer.zero_grad()

                    if configs.uses_cnn:
                        utterance_batch, utterance_len_batch, flat_label_seq_batch, label_seq_len_batch = batch
                        # [max_utterance_len, batch_size, class_num], [batch_size]
                        output_batch, features_seq_len_batch = self.model.forward(utterance_batch, utterance_len_batch)
                        # [batch_size, max_utterance_len]
                        best_prediction_seq_batch = self.decoder(
                            output_batch, features_seq_len_batch
                        )
                    else:
                        utterance_batch, flat_label_seq_batch, label_seq_len_batch = batch
                        # [max_utterance_len, batch_size, class_num], [batch_size]
                        output_batch, utterance_len_batch = self.model.forward(utterance_batch)
                        # [batch_size, max_utterance_len]
                        best_prediction_seq_batch = self.decoder(
                            output_batch, utterance_len_batch
                        )

                    avg_dist += self.calc_avg_dist(
                        best_prediction_seq_batch,
                        flat_label_seq_batch, label_seq_len_batch
                    )

                    if pct >= next_logging_pct:
                        log(
                            f'{int(pct)}%, time: {time.time() - start_time}, '
                            f'avg_dist: {avg_dist / batch_num}'
                        )
                        next_logging_pct += 10.

                avg_dist /= batch_num
                self.lr_scheduler.step(avg_dist)

                log(
                    f'avg_dev_time: {time.time() - start_time}\n'
                    f'avg_dist: {avg_dist}'
                )

                if avg_dist < self.min_avg_dist:
                    self.min_avg_dist = avg_dist
                    self.save_ckpt()

    def test(self):
        if configs.tests_ensemble:
            self.test_ensemble()
            return

        with torch.no_grad():
            log('testing')
            self.load_ckpt()
            self.model.eval()
            next_logging_pct = 10.
            best_prediction_seqs = [[] for idx in range(data_utils.get_data_size('test'))]
            start_time = time.time()

            for pct, batch in data_utils.gen_batches('test'):
                if configs.uses_cnn:
                    utterance_batch, utterance_len_batch, idx_batch = batch
                    # [max_utterance_len, batch_size, class_num], [batch_size]
                    output_batch, features_seq_len_batch = self.model.forward(utterance_batch, utterance_len_batch)
                    # [batch_size, max_utterance_len]
                    best_prediction_seq_batch = self.decoder(
                        output_batch, features_seq_len_batch
                    )
                else:
                    utterance_batch, idx_batch = batch
                    # [max_utterance_len, batch_size, class_num], [batch_size]
                    output_batch, utterance_len_batch = self.model.forward(utterance_batch)
                    # [batch_size, max_utterance_len]
                    best_prediction_seq_batch = self.decoder(
                        output_batch, utterance_len_batch
                    )

                for idx, best_prediction_seq in zip(idx_batch, best_prediction_seq_batch):
                    best_prediction_seqs[idx] = best_prediction_seq

                if pct >= next_logging_pct:
                    log(
                        f'{int(pct)}%, time: {time.time() - start_time}'
                    )
                    next_logging_pct += 10.

            data_utils.save_submission(best_prediction_seqs)

    def test_ensemble(self):
        assert configs.uses_cnn

        with torch.no_grad():
            log('testing ensemble')

            ckpt_paths = sorted(
                [path for path in os.listdir(f'{configs.ckpts_dir}/') if path.endswith('.ckpt')],
                key=Runner.to_timestamp_and_epoch_idx
            )
            ckpt_paths = [f'{configs.ckpts_dir}/{ckpt_path}' for ckpt_path in ckpt_paths][-10:]
            avg_output_batches, features_seq_len_batches, idx_batches = [], [], []
            model_num = 0
            best_prediction_seqs = [[] for idx in range(data_utils.get_data_size('test'))]

            for ckpt_path in ckpt_paths:
                self.load_ckpt(ckpt_path)
                self.model.eval()
                next_logging_pct = 10.
                start_time = time.time()

                for i, (pct, batch) in enumerate(data_utils.gen_batches('test')):
                    utterance_batch, utterance_len_batch, idx_batch = batch
                    # [max_utterance_len, batch_size, class_num], [batch_size]
                    output_batch, features_seq_len_batch = self.model.forward(utterance_batch, utterance_len_batch)

                    if not model_num:
                        avg_output_batches.append(output_batch.detach().cpu())
                        features_seq_len_batches.append(features_seq_len_batch)
                        idx_batches.append(idx_batch)
                    else:
                        avg_output_batches[i] += output_batch.detach().cpu()

                    if pct >= next_logging_pct:
                        log(
                            f'{int(pct)}%, time: {time.time() - start_time}'
                        )
                        next_logging_pct += 10.

                model_num += 1

            log('testing on average outputs')

            for output_batch, features_seq_len_batch, idx_batch in zip(
                    avg_output_batches, features_seq_len_batches, idx_batches
            ):
                avg_output_batch = output_batch / model_num

                # [batch_size, max_utterance_len]
                best_prediction_seq_batch = self.decoder(
                    avg_output_batch, features_seq_len_batch
                )

                for idx, best_prediction_seq in zip(idx_batch, best_prediction_seq_batch):
                    best_prediction_seqs[idx] = best_prediction_seq

            data_utils.save_submission(best_prediction_seqs)

    def get_ckpt(self):
        return {
            'epoch_idx': self.epoch_idx,
            'min_avg_dist': self.min_avg_dist,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'criterion': self.ctc_loss.state_dict()
        }

    def set_ckpt(self, ckpt):
        self.epoch_idx = ckpt['epoch_idx'] + 1

        if not configs.uses_new_classifier:
            self.min_avg_dist = ckpt['min_avg_dist']
            self.lr_scheduler.step(self.min_avg_dist)
        # self.model.load_state_dict(ckpt['model'])

        # in case we remove some modules
        model_state_dict = self.model.state_dict()

        if configs.training and configs.uses_new_classifier:
            classifier_param_names = [name for name in ckpt['model'] if 'classifier' in name]

            for name in classifier_param_names:
                del ckpt['model'][name]

        if configs.training and configs.uses_new_transformer:
            rnn_param_names = [name for name in ckpt['model'] if 'transformer' in name]

            for name in rnn_param_names:
                del ckpt['model'][name]

        if configs.training and configs.uses_new_encoder:
            rnn_param_names = [name for name in ckpt['model'] if 'encoder' in name]

            for name in rnn_param_names:
                del ckpt['model'][name]

        model_state_dict.update(
            {
                param_name: value
                for param_name, value in ckpt['model'].items()
                if param_name in model_state_dict
            }
        )
        self.model.load_state_dict(model_state_dict)
        # self.model.load_state_dict(ckpt['model'])

        if configs.loads_optimizer_state:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            print('loaded optimizer')

        # self.criterion.load_state_dict(ckpt['criterion'])

    ckpt = property(get_ckpt, set_ckpt)

    def save_ckpt(self):
        ckpt_path = f'{configs.ckpts_dir}/{configs.timestamp}.{self.epoch_idx}.ckpt'
        log(f'saving checkpoint {ckpt_path}')
        torch.save(self.ckpt, f=ckpt_path)

    @staticmethod
    def to_timestamp_and_epoch_idx(ckpt_path_):
        date, time, epoch_idx = map(int, re.split(r'[-.]', ckpt_path_[:ckpt_path_.find('.ckpt')]))
        return date, time, epoch_idx

    def load_ckpt(self, ckpt_path=None):
        if not ckpt_path:
            if configs.ckpt_id:
                ckpt_path = f'{configs.ckpts_dir}/{configs.ckpt_id}.ckpt'
            elif configs.loads_best_ckpt:
                ckpt_path = '{configs.ckpts_dir}/ckpt.best'
            else:
                ckpt_paths = [path for path in os.listdir(f'{configs.ckpts_dir}/') if path.endswith('.ckpt')]
                ckpt_path = f'{configs.ckpts_dir}/{sorted(ckpt_paths, key=Runner.to_timestamp_and_epoch_idx)[-1]}'

        print(f'loading checkpoint {ckpt_path}')

        self.ckpt = torch.load(ckpt_path)


if __name__ == '__main__':
    runner = Runner()

    if configs.training:
        runner.train()
    else:
        runner.test()
