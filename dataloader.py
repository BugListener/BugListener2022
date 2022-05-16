import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import pickle, pandas as pd
import json


class dialogDataset(Dataset):
    def __init__(self, file_name, tokenizer_address='bert-base-uncased', max_length=80):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_address)

        with open(file_name, 'r') as r:
            data_dict = json.load(r)

        self.ids = data_dict['ids']
        self.dialog = data_dict['dialog']
        self.role = data_dict['role']
        self.label = data_dict['label']
        self.edge = data_dict['edge']
        self.len = len(self.ids)
        self.max_length = max_length

    def __getitem__(self, index):
        return self.dialog[index], \
               torch.tensor(self.label[index]), \
               self.ids[index], \
               self.role[index], \
               self.edge[index], \
               len(self.dialog[index])

    def __len__(self):
        return self.len

    def tokenize_in_minibatch(self, dialog_list):
        # 对文本进行id的映射
        dialog_input_ids = []
        dialog_token_type_ids = []
        dialog_attention_mask = []
        for dialog in dialog_list:
            tokenized_input = self.tokenizer(dialog, padding="max_length", truncation=True,
                                             return_tensors="pt", max_length=self.max_length)
            dialog_input_ids.append(tokenized_input['input_ids'])
            dialog_token_type_ids.append(tokenized_input['token_type_ids'])
            dialog_attention_mask.append(tokenized_input['attention_mask'])

        assert len(dialog_list) == len(dialog_input_ids) == len(dialog_token_type_ids) == len(dialog_attention_mask)
        return [dialog_input_ids, dialog_token_type_ids, dialog_attention_mask]

    # dataloader自定义padding操作
    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return_list = []
        for i in dat:
            if i == 0:
                for token in self.tokenize_in_minibatch(dat[i].tolist()):
                    return_list.append(pad_sequence(token, True))
            elif i == 1:
                return_list.append(torch.tensor(dat[i]))
            else:
                return_list.append(dat[i].tolist())

        return return_list
