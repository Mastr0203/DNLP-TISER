import torch
from torch.utils.data import Dataset
import json

import torch
from torch.utils.data import Dataset
import json


class FinetuningDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        """
        data_path: paht to file json / jsonl
        tokenizer: tokenizer of the model to finetune
        """
        self.tokenizer = tokenizer
        self.data = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): #skip empty lines
                    obj = json.loads(line)
                    self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        output = item['output']

        # build the struct of the chat
        messages = [
            {"role": "prompt", "content": prompt},
            {"role": "output", "content": output}
        ]

        # apply chat template
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        return {"text": formatted_text}