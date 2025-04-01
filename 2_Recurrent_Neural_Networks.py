#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
from sklearn.metrics import classification_report, roc_auc_score

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
# In[2]:


seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# In[3]:


train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])


# In[4]:


tokenizer = torchtext.data.utils.get_tokenizer("basic_english")


# In[5]:


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    length = len(tokens)
    return {"tokens": tokens, "length": length}


# In[6]:


max_length = 256

train_data = train_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)
test_data = test_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)


# In[7]:


test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]


# In[8]:


min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)


# In[9]:


unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]


# In[10]:


vocab.set_default_index(unk_index)


# In[11]:


def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}


# In[12]:


train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})


# In[13]:


train_data = train_data.with_format(type="torch", columns=["ids", "label", "length"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label", "length"])
test_data = test_data.with_format(type="torch", columns=["ids", "label", "length"])


# In[14]:


train_data[0]


# In[15]:


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_length = [i["length"] for i in batch]
        batch_length = torch.stack(batch_length)
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "length": batch_length, "label": batch_label}
        return batch

    return collate_fn


# In[16]:


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


# In[17]:


batch_size = 256

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


# In[18]:


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout_rate,
        pad_index,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate,
            batch_first=True,
        )
        # self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, output_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, length, batch_first=True, enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = self.dropout(hidden[-1])
            # hidden = [batch size, hidden dim]
        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction


# In[19]:
vocab_size = len(vocab)
embedding_dim = 768    # 原为300
hidden_dim = 1024      # 原为300
output_dim = len(train_data.unique("label"))
n_layers = 4           # 原为2
bidirectional = True
dropout_rate = 0.6

model = LSTM(
    vocab_size,
    embedding_dim,
    hidden_dim,
    output_dim,
    n_layers,
    bidirectional,
    dropout_rate,
    pad_index,
)


# In[20]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


# In[21]:


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                nn.init.orthogonal_(param)


# In[22]:


model.apply(initialize_weights)


# # In[24]:
#
#
# vectors = torchtext.vocab.GloVe()
#
#
# # In[25]:
#
#
# pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
#
#
# # In[26]:
#
#
# model.embedding.weight.data = pretrained_embedding


# In[27]:


lr = 5e-4

optimizer = optim.Adam(model.parameters(), lr=lr)


# In[28]:


criterion = nn.CrossEntropyLoss()


# In[29]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device


# In[30]:


model = model.to(device)
criterion = criterion.to(device)


# In[31]:


def train(dataloader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(dataloader, desc="training..."):
        ids = batch["ids"].to(device)
        length = batch["length"]
        label = batch["label"].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


# In[32]:


# 修改 In[32] 单元格的evaluate函数：


def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            length = batch["length"]
            label = batch["label"].to(device)
            prediction = model(ids, length)

            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)

            # 收集预测结果
            probs = torch.softmax(prediction, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(label.cpu().numpy())

            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

    # 合并所有batch结果
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 计算指标
    report = classification_report(
        all_labels,
        all_preds.argmax(axis=1),
        target_names=['negative', 'positive'],
        output_dict=True
    )
    auroc = roc_auc_score(all_labels, all_preds[:, 1])

    return (
        np.mean(epoch_losses),
        np.mean(epoch_accs),
        report['macro avg']['precision'],
        report['macro avg']['recall'],
        report['macro avg']['f1-score'],
        auroc
    )


# In[33]:


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


# In[34]:


n_epochs = 10
best_valid_loss = float("inf")

# 修改 In[34] 单元格的训练循环：
metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, device)
    valid_loss, valid_acc, valid_prec, valid_rec, valid_f1, valid_auc = evaluate(valid_data_loader, model, criterion, device)

    # 记录所有指标
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    metrics['valid_precisions'].append(valid_prec)
    metrics['valid_recalls'].append(valid_rec)
    metrics['valid_f1s'].append(valid_f1)
    metrics['valid_aucs'].append(valid_auc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "lstm.pt")

    # 打印结果
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")
    print(f"Epoch {epoch}: peak GPU memory {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")


# In[35]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="train loss")
ax.plot(metrics["valid_losses"], label="valid loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()


# In[36]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()


# In[37]:


model.load_state_dict(torch.load("lstm.pt"))

test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
    test_data_loader, model, criterion, device
)


# In[38]:

print(f"Test loss: {test_loss:.3f}, Test acc: {test_acc:.3f}")
print(f"Test Precision: {test_prec:.3f}")
print(f"Test Recall: {test_rec:.3f}")
print(f"Test F1: {test_f1:.3f}")
print(f"Test AUC: {test_auc:.3f}")


# In[39]:


def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    length = torch.LongTensor([len(ids)])
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor, length).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability


# In[40]:


text = "This film is terrible!"

predict_sentiment(text, model, tokenizer, vocab, device)


# In[41]:


text = "This film is great!"

predict_sentiment(text, model, tokenizer, vocab, device)


# In[42]:


text = "This film is not terrible, it's great!"

predict_sentiment(text, model, tokenizer, vocab, device)
predict_sentiment(text, model, tokenizer, vocab, device)


# In[43]:


text = "This film is not great, it's terrible!"


