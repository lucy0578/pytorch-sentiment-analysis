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
import transformers
from sklearn.metrics import classification_report, roc_auc_score


# In[2]:


seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# In[3]:


train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])


# In[4]:


transformer_name = "bert-base-uncased"

tokenizer = transformers.AutoTokenizer.from_pretrained(transformer_name)


# In[5]:


tokenizer.tokenize("hello world!")


# In[6]:


tokenizer.encode("hello world!")


# In[7]:


tokenizer.convert_ids_to_tokens(tokenizer.encode("hello world"))


# In[8]:


tokenizer("hello world!")


# In[9]:


def tokenize_and_numericalize_example(example, tokenizer):
    ids = tokenizer(example["text"], truncation=True)["input_ids"]
    return {"ids": ids}


# In[10]:


train_data = train_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)
test_data = test_data.map(
    tokenize_and_numericalize_example, fn_kwargs={"tokenizer": tokenizer}
)


# In[11]:


train_data[0]


# In[12]:


tokenizer.vocab["!"]


# In[13]:


tokenizer.pad_token


# In[14]:


tokenizer.pad_token_id


# In[15]:


tokenizer.vocab[tokenizer.pad_token]


# In[16]:


pad_index = tokenizer.pad_token_id


# In[17]:


test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]


# In[18]:


train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])


# In[19]:


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_ids = [i["ids"] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(
            batch_ids, padding_value=pad_index, batch_first=True
        )
        batch_label = [i["label"] for i in batch]
        batch_label = torch.stack(batch_label)
        batch = {"ids": batch_ids, "label": batch_label}
        return batch

    return collate_fn


# In[20]:


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


# In[21]:


batch_size = 8

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


# In[22]:


class Transformer(nn.Module):
    def __init__(self, transformer, output_dim, freeze):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.config.hidden_size
        self.fc = nn.Linear(hidden_dim, output_dim)
        if freeze:
            for param in self.transformer.parameters():
                param.requires_grad = False

    def forward(self, ids):
        # ids = [batch size, seq len]
        output = self.transformer(ids, output_attentions=True)
        hidden = output.last_hidden_state
        # hidden = [batch size, seq len, hidden dim]
        attention = output.attentions[-1]
        # attention = [batch size, n heads, seq len, seq len]
        cls_hidden = hidden[:, 0, :]
        prediction = self.fc(torch.tanh(cls_hidden))
        # prediction = [batch size, output dim]
        return prediction


# In[23]:


transformer = transformers.AutoModel.from_pretrained(transformer_name)


# In[24]:


transformer.config.hidden_size


# In[25]:


output_dim = len(train_data["label"].unique())
freeze = False

model = Transformer(transformer, output_dim, freeze)


# In[26]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


# In[27]:


lr = 1e-5

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


def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    for batch in tqdm.tqdm(data_loader, desc="training..."):
        ids = batch["ids"].to(device)
        label = batch["label"].to(device)
        prediction = model(ids)
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
from sklearn.metrics import classification_report, roc_auc_score


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


n_epochs = 3
best_valid_loss = float("inf")

# 修改 In[34] 单元格的训练循环：
metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, device)
    valid_loss, valid_acc, valid_prec, valid_rec, valid_f1, valid_auc = evaluate(valid_data_loader, model, criterion, device)

    # 记录所有指标
    metrics['train_losses'].append(train_loss)
    metrics['train_accs'].append(train_acc)
    metrics['valid_losses'].append(valid_loss)
    metrics['valid_accs'].append(valid_acc)
    metrics['valid_precisions'].append(valid_prec)
    metrics['valid_recalls'].append(valid_rec)
    metrics['valid_f1s'].append(valid_f1)
    metrics['valid_aucs'].append(valid_auc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "transformer.pt")

    # 打印结果
    print(f"valid_precision: {valid_prec:.3f}")
    print(f"valid_recall: {valid_rec:.3f}")
    print(f"valid_f1: {valid_f1:.3f}")
    print(f"valid_auc: {valid_auc:.3f}")


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
plt.show()


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
plt.show()


# In[38]:


model.load_state_dict(torch.load("transformer.pt"))

test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
    test_data_loader, model, criterion, device
)


# In[39]:


print(f"Test Precision: {test_prec:.3f}")
print(f"Test Recall: {test_rec:.3f}")
print(f"Test F1: {test_f1:.3f}")
print(f"Test AUC: {test_auc:.3f}")


# In[40]:


def predict_sentiment(text, model, tokenizer, device):
    ids = tokenizer(text)["input_ids"]
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability


# In[41]:


text = "This film is terrible!"

predict_sentiment(text, model, tokenizer, device)


# In[42]:


text = "This film is great!"

predict_sentiment(text, model, tokenizer, device)


# In[43]:


text = "This film is not terrible, it's great!"

predict_sentiment(text, model, tokenizer, device)


# In[44]:


text = "This film is not great, it's terrible!"

predict_sentiment(text, model, tokenizer, device)

