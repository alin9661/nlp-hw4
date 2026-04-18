import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.encoder_inputs = []
        self.encoder_masks = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.is_test = (split == "test")

        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_lines = load_lines(nl_path)

        # Load SQL queries (not available for test set)
        sql_lines = None
        if not self.is_test:
            sql_path = os.path.join(data_folder, f'{split}.sql')
            sql_lines = load_lines(sql_path)

        # The decoder_start_token_id for T5 is the pad token (0)
        decoder_start_id = tokenizer.pad_token_id

        for i in range(len(nl_lines)):
            # Encoder: prefix + natural language query
            input_text = "translate English to SQL: " + nl_lines[i]
            enc = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512)
            enc_ids = enc['input_ids'].squeeze(0)       # (T,)
            enc_mask = enc['attention_mask'].squeeze(0)  # (T,)

            self.encoder_inputs.append(enc_ids)
            self.encoder_masks.append(enc_mask)

            if not self.is_test:
                # Decoder target: SQL query tokens + EOS (already appended by tokenizer)
                target_enc = tokenizer(sql_lines[i], return_tensors='pt', truncation=True, max_length=512)
                target_ids = target_enc['input_ids'].squeeze(0)  # (T',) — ends with EOS token

                # Decoder input: shift right — prepend decoder_start_id, drop last token of targets
                dec_input = torch.cat([torch.tensor([decoder_start_id]), target_ids[:-1]])

                self.decoder_inputs.append(dec_input)
                self.decoder_targets.append(target_ids)

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        if self.is_test:
            return (
                self.encoder_inputs[idx],
                self.encoder_masks[idx],
            )
        else:
            return (
                self.encoder_inputs[idx],
                self.encoder_masks[idx],
                self.decoder_inputs[idx],
                self.decoder_targets[idx],
            )

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item[0] for item in batch]
    encoder_mask_list = [item[1] for item in batch]
    decoder_input_list = [item[2] for item in batch]
    decoder_target_list = [item[3] for item in batch]

    # Pad sequences (PAD_IDX = 0, which is T5's pad token)
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)
    decoder_inputs = pad_sequence(decoder_input_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_target_list, batch_first=True, padding_value=PAD_IDX)

    # Initial decoder input: first token of each decoder input sequence (decoder_start_token_id)
    initial_decoder_inputs = decoder_inputs[:, 0:1]  # (B, 1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = [item[0] for item in batch]
    encoder_mask_list = [item[1] for item in batch]

    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_mask_list, batch_first=True, padding_value=0)

    # Initial decoder input: pad_token_id (decoder_start_token_id for T5)
    initial_decoder_inputs = torch.full((encoder_ids.size(0), 1), PAD_IDX, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x
