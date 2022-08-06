import time

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import BertTokenizer, BertModel

my_tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')

my_model_pretrained = BertModel.from_pretrained('./bert-base-chinese')


def collate_fn1(data):
    texts = [i['text'] for i in data]
    labels = [i['label'] for i in data]

    data = my_tokenizer.batch_encode_plus(batch_text_or_text_pairs=texts, truncation=True, max_length=500,
                                          padding='max_length', return_length=True, return_tensors='pt')

    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    return input_ids, attention_mask, token_type_ids, labels


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = my_model_pretrained(input_ids=input_ids, attention_mask=attention_mask,
                                      token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
        return out


def train_model():
    my_model = MyModel()
    my_optim = AdamW(my_model.parameters(), lr=5e-4)
    my_cross = nn.CrossEntropyLoss()

    my_dataset = load_dataset('csv', data_files='./mydata1/train.csv', split="train")
    for param in my_model_pretrained.parameters():
        param.requires_grad_(False)

    epochs = 3
    for epoch_idx in range(epochs):
        starttime = time.time()
        my_dataloader = DataLoader(my_dataset, batch_size=8, collate_fn=collate_fn1, shuffle=False, drop_last=True)
        for i, (input_ids, attention_mask, token_type_ids, labels) in enumerate(my_dataloader):
            my_out = my_model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
            my_loss = my_cross(my_out, labels)
            my_optim.zero_grad()
            my_loss.backward()
            my_optim.step()
            if i % 5 == 0:
                out = my_out.argmax(dim=1)  # [8,2] --> (8,)
                accuracy = (out == labels).sum().item() / len(labels)
                print('轮次:%d 迭代数:%d 损失:%.6f 准确率%.3f 时间%d'
                      % (epoch_idx, i, my_loss.item(), accuracy, (int)(time.time()) - starttime))
        torch.save(my_model.state_dict(), './my_model_%d.bin' % (epoch_idx + 1))


def evaluate_model():
    my_dataset_test = load_dataset('csv', data_files='./mydata1/test.csv', split='train')

    path = './my_model_1.bin'
    my_model = MyModel()
    my_model.load_state_dict(torch.load(path))
    my_model.eval()
    correct = 0
    total = 0
    my_loader_test = torch.utils.data.DataLoader(my_dataset_test,
                                                 batch_size=8,
                                                 collate_fn=collate_fn1,
                                                 shuffle=True,
                                                 drop_last=True)

    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(my_loader_test):

        with torch.no_grad():
            my_out = my_model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids)

        out = my_out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)
        if i % 5 == 0:
            print(correct / total, end=" ")
            print(my_tokenizer.decode(input_ids[0], skip_special_tokens=True), end=" ")
            print('预测值 真实值:', out[0].item(), labels[0].item())


if __name__ == '__main__':
    train_model()
    # evaluate_model()
