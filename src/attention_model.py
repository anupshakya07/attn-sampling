

from torchtext.data import Field
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gensim
import torch.optim as optim
import numpy as np
import pickle
import collections
import math
import os
import time


class MathiaDataset(Dataset):

    def __init__(self, data_path, embedding_model_path, dataset_type="train"):
        self.data = pickle.load(open(data_path, "rb"))
        # self.data["train"] = self.data["train"][230809:]
        # self.data["val"] = self.data["val"][0:1000]
        # self.data["train"] = random.sample(self.data["train"], 100000)
        # self.data["val"] = random.sample(self.data["val"], 1000)
        self.max_length = self.data["max_length"]
        self.embedding_model = gensim.models.Word2Vec.load(embedding_model_path)
        self.one_hot_encoder = one_hot_encoder([1, 0])
        self.kc_vocab_size = len(self.data['kc_index_map'])
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data[self.dataset_type])

    def set_max_seq_length(self, min_length, max_length):
        self.max_length = max_length if max_length > self.max_length else self.max_length
        data = []
        for inst in self.data[self.dataset_type]:
            if max_length > len(inst["src"]) > min_length:
                data.append({"src": inst["src"], "target": inst["target"]})
        self.data[self.dataset_type] = data

    def __getitem__(self, idx):
        src = self.data[self.dataset_type][idx]["src"]
        trg = self.data[self.dataset_type][idx]["target"]
        inst_length = len(src)

        # src_op = [self.embedding_model.wv[op] for op in src]
        # trg_op = [self.one_hot_encoder.transform(op) for op in trg]
        # trg_op = self.one_hot_encoder.transform(np.array(trg).reshape(-1, 1))
        return {"src": src, "target": trg, "length": inst_length}


def collate_data(batch):
    element = batch[0]

    if isinstance(element, collections.abc.Mapping):
        return {key: collate_data([d[key] for d in batch]) for key in element}

    elif isinstance(element, collections.abc.Sequence):
        it = iter(batch)
        # element_size = len(next(it))
        data_length_array = np.array([len(element) for element in it])
        max_len = data_length_array.max()
        for element in batch:
            element_len = len(element)
            padding_length = max_len - element_len
            padding = [2] * padding_length
            element.extend(padding)
        # if not all(len(el) == element_size for el in it):
        #     raise RuntimeError('each element in list of batch should be of equal size')
        return [collate_data(samples) for samples in batch]

    elif isinstance(element, int):
        return torch.tensor(batch, dtype=torch.int64)


def one_hot_encoder(vocab):
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(np.array(vocab).reshape(-1, 1))
    return encoder


def train(model, training_data, validation_data, optimizer, device, opt):
    def print_performances(header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", ppl=ppl,
            accu=100 * accu, elapse=(time.time() - start_time) / 60, lr=lr))

    model = model.to(device)

    log_train_file = os.path.join(opt.output_dir, 'train.log')
    log_valid_file = os.path.join(opt.output_dir, 'valid.log')

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,ppl,accuracy\n')
        log_vf.write('epoch,loss,ppl,accuracy\n')

    op_file = open("output/attention_scores_new_2022.txt", "w")
    valid_losses = []

    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        final_epoch = epoch_i == (opt.epoch - 1)

        start = time.time()
        train_loss, train_accu = train_epoch(model, training_data, optimizer, opt, device,
                                             smoothing=opt.label_smoothing,
                                             output_file=op_file,
                                             final_epoch=final_epoch)
        train_ppl = math.exp(min(train_loss, 100))
        lr = optimizer._optimizer.param_groups[0]['lr']
        print_performances('Training', train_ppl, train_accu, start, lr)

        valid_loss, valid_accu = eval_epoch(model, validation_data, device, opt)
        valid_ppl = math.exp(min(valid_loss, 100))
        print_performances('Validation', valid_ppl, valid_accu, start, lr)

        valid_losses += [valid_loss]

        checkpoint = {'epoch': epoch_i, 'settings': opt, 'model': model.state_dict()}
        if opt.save_mode == 'all':
            model_name = 'model_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
            torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
        elif opt.save_mode == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(opt.output_dir, model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=train_loss,
                ppl=train_ppl, accu=100 * train_accu))
            log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                ppl=valid_ppl, accu=100 * valid_accu))

    op_file.close()


def train_epoch(model, training_data, optimizer, opt, device, smoothing, output_file, final_epoch=False):
    model.train()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Training)   '
    processed_attentions = dict()
    for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
        # for src, target in zip(batch["src"], batch["target"]):
        max_length = batch["length"].max().item()
        src_seq = torch.zeros([opt.batch_size, max_length], dtype=torch.int64)
        target_seq = torch.zeros([opt.batch_size, max_length], dtype=torch.int64)
        torch.cat(batch["src"], out=src_seq, dim=0)
        torch.cat(batch["target"], out=target_seq, dim=0)
        src_seq = src_seq.view(opt.batch_size, max_length)
        target_seq = target_seq.view(opt.batch_size, max_length)
        src_seq = src_seq.to(device)
        target_seq, gold = map(lambda x: x.to(device), patch_trg(target_seq, 2))

        optimizer.zero_grad()
        pred, dec_enc_attn = model(src_seq, target_seq, return_attns=final_epoch)
        if dec_enc_attn:
            processed_batch_attentions = process_attentions(dec_enc_attn, src_seq, output_file,
                                                            training_data.dataset.kc_vocab_size)
            processed_attentions.update(processed_batch_attentions)
        loss, n_correct, n_word = cal_performance(pred, gold, 2, smoothing=smoothing)
        loss.backward()
        optimizer.step_and_update_lr()

        # note keeping
        n_word_total += n_word
        n_word_correct += n_correct
        total_loss += loss.item()
    pickle.dump(processed_attentions, open("output/processed_attention_scores.pkl", "wb"))

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device, opt):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss, n_word_total, n_word_correct = 0, 0, 0

    desc = '  - (Validation) '
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=desc, leave=False):
            max_length = batch["length"].max().item()
            # prepare data
            src_seq = torch.zeros([opt.batch_size, max_length], dtype=torch.int64)
            target_seq = torch.zeros([opt.batch_size, max_length], dtype=torch.int64)
            torch.cat(batch["src"], out=src_seq, dim=0)
            torch.cat(batch["target"], out=target_seq, dim=0)
            src_seq = src_seq.view(opt.batch_size, max_length)
            target_seq = target_seq.view(opt.batch_size, max_length)
            src_seq = src_seq.to(device)
            target_seq, gold = map(lambda x: x.to(device), patch_trg(target_seq, 2))

            # forward
            pred, _ = model(src_seq, target_seq, return_attns=False)
            loss, n_correct, n_word = cal_performance(
                pred, gold, 2, smoothing=False)

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def cal_performance(pred, gold, trg_pad_idx, smoothing=False):
    ''' Apply label smoothing if needed '''

    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(trg_pad_idx)
    n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()

    return loss, n_correct, n_word


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    # trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold


def process_attentions(attentions, src_seq, op_file, kc_vocab_size):
    # Averaging the attentions over the 6 layers
    average_attentions = np.mean([x.transpose(1, 2).cpu().detach().numpy() for x in attentions], axis=0)
    src_np = src_seq.cpu().numpy()
    processed_attentions = dict()
    for index, instance in enumerate(src_np):
        # 2 is the masking key
        # 3 is the Beginning of sentence token
        # 4 is the End of Sentence token
        # to count number of elements not equal to 2, 3 and 4; subtract 2 as compensation for tokens 3 and 4
        actual_kcs_count = np.count_nonzero(instance != 2) - 2
        temp_np = np.zeros(kc_vocab_size + 5, dtype=float)
        inst_attention = []
        for i in range(actual_kcs_count):
            kc_attn = average_attentions[index][i + 1].mean()
            # temp_np[instance[i+1]] += kc_attn
            inst_attention.append(kc_attn)
        src_str_list = [str(ist) for ist in np.delete(instance, np.where(instance == 2)).tolist()]
        src_str_key = ",".join(src_str_list)
        processed_attentions[src_str_key] = inst_attention
    return processed_attentions
