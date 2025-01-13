#!/usr/bin/env python
# coding: utf-8

# # 1 - NBoW
# 
# In this series we'll be building a machine learning model to perform sentiment analysis -- a subset of text classification where the task is to detect if a given sentence is positive or negative -- using [PyTorch](https://github.com/pytorch/pytorch) and [torchtext](https://github.com/pytorch/text). The dataset used will be movie reviews from the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/), which we'll obtain using the [datasets](https://github.com/huggingface/datasets) library.
# 

# ## Introduction
# 
# In this first notebook, we'll start very simple with one of the most basic models for _NLP_ (natural language processing): a _NBoW_ (_neural bag-of-words_) model (also known as _continuous bag-of-words_, _CBoW_). The NBoW model are a strong, commonly used, baseline model for NLP tasks. They should be one of the first models you implement when performing sentiment analysis/text classification.
# 
# ![](assets/nbow_model.png)
# 
# An NBoW model takes in a sequence of $T$ _tokens_, $X=\{x_1,...,x_T\} \in \mathbb{Z}^T$ and passes each token through an _embedding layer_ to obtain a sequence of _embedding vectors_. The sequence of embedding vectors is just known as an _embedding_, $E=\{e_1,...,e_T\} \in \mathbb{R}^{T \times D}$, where $D$ is known as the _embedding dimension_. It then _pools_ the embeddings across the sequence dimension to get $P \in \mathbb{R}^D$ and then finally passes $P$ through a linear layer (also known as a fully connected layer), to get a prediction, $\hat{Y} \in \mathbb{R}^C$, where $C$ is the number of classes. We'll explain what a token is, and what each of the layers -- embedding layer, pooling, and linear layer -- do in due course.
# 
# A note on notation, what does something like $E=\{e_1,...,e_T\} \in \mathbb{R}^{T \times D}$ mean? $\mathbb{R}^{T \times D}$ means a $T \times D$ sized tensor full of real numbers, i.e. a `torch.FloatTensor`. $X=\{x_1,...,x_T\} \in \mathbb{Z}^T$ is a $T$ sized tensor full of integers, i.e. a `torch.LongTensor`.
# 

# ## Preparing Data
# 
# Before we can implement our NBoW model, we first have to perform quite a few steps to get our data ready to use. NLP usually requires quite a lot of data wrangling beforehand, though libraries such as `datasets` and `torchtext` handle most of this for us.
# 
# The steps to take are:
# 
# -   importing modules
# -   loading data
# -   tokenizing data
# -   creating data splits
# -   creating a vocabulary
# -   numericalizing data
# -   creating the data loaders
# 

# ### Importing Modules
# 
# First, we'll import the required modules.
# 
# We use the `datasets` module for handling datasets, `matplotlib` for plotting our results, `numpy` for numerical processing, `torch` for tensor computations, `torch.nn` for neural networks, `torch.optim` for neural network optimizers, `torchtext` for text processing, and `tqdm` for measuring progress.
# 

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


# We'll also make sure to set the random seeds for `torch` and `numpy`. This is to ensure this notebook is reproducable, i.e. we get the same results each time we run it.
# 
# It is usually good practice to run your experiments multiple times with different random seeds -- both to measure the variance of your model and also to avoid having results only calculated with either "good" or "bad" seeds, i.e. being very lucky or unlucky with the randomness in the training process.
# 

# In[2]:


seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# ### Loading the Dataset
# 
# Next, we'll load our dataset using the `datasets` library. The first argument is the name of the dataset and the `split` argument chooses which _splits_ of the data we want.
# 
# Datasets usually come in two or more _splits_, non-overlapping examples from the data, most commonly a _train split_ -- which we train our model on -- and a _test split_ -- which we evaluate our trained model on. There's also a _validation split_, which we'll talk more about later. The train, test and validation split are also commonly called the train, test and validation sets -- we'll use split and set interchangeably
# in these tutorials -- and the dataset usually refers to all three of the sets combined. The IMDb dataset actually comes with a third split, called _unsupervised_, which contains a bunch of examples without labels. We don't want these so we don't include them in our `split` argument. Note that if we didn't pass an argument to `split` then it would load all available splits of the data.
# 
# How do we know that we have to use "imdb" for the IMDb dataset and that there's an "unsupervised" split? The `datasets` library has a great website used to browse the available datasets, see: https://huggingface.co/datasets/. By navigating to the [IMDb dataset page](https://huggingface.co/datasets/imdb) we can see more information specifically about the IMDb dataset.
# 
# The output received when loading the dataset tells us that it is using a locally cached version instead of downloading the dataset from online.
# 

# In[3]:


train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])


# We can print out the splits which shows us the _features_ and _num_rows_ of the dataset. num*rows are the number of examples in split, as we can see, there are 25,000 examples in each. Each example in a dataset provided by the `datasets` library is a dictionary, and the features are the keys which appear in every one of those dictionaries/examples. So, each example in the IMDb dataset has a \_text* and a _label_ key.
# 

# In[4]:


train_data, test_data


# We can check the `features` attribute of a split to get more information about the features. We can see that _text_ is a `Value` of `dtype=string` -- in other words, it's a string -- and that _label_ is a `ClassLabel`. A `ClassLabel` means the feature is an integer representation of which class the example belongs to. `num_classes=2` means that our labels are one of two values, 0 or 1, and `names=['neg', 'pos']` gives us the human-readable versions of those values. Thus, a label of 0 means the example is a negative review and a label of 1 means the example is a positive review.
# 

# In[5]:


train_data.features


# We can look at an example by indexing into the train set. As we can see, the text is quite noisy and also rambles on quite a bit.
# 

# In[6]:


train_data[0]


# ### Tokenization
# 
# One of the first things we need to do to our data is _tokenize_ it. Machine learning models aren't designed to handle strings, they're design to handle numbers. So what we need to do is break down our string into individual _tokens_, and then convert these tokens to numbers. We'll get to the conversion later, but first we'll look at _tokenization_.
# 
# Tokenization involves using a _tokenizer_ to process the strings in our dataset. A tokenizer is a function that goes from a string to a list of strings. There are many types of tokenizers available, but we're going to use a relatively simple one provided by `torchtext` called the `basic_english` tokenizer. We load our tokenizer as such:
# 

# In[7]:


tokenizer = torchtext.data.utils.get_tokenizer("basic_english")


# We can use the tokenizer by calling it on a string.
# 
# Notice it creates a token by splitting the word on spaces, separating punctuation into its own token, and also lowercasing every word.
# 
# The `get_tokenizer` function also supports other tokenizers, such as ones provided by [spaCy](https://spacy.io/) and [nltk](https://www.nltk.org/).
# 

# In[8]:


tokenizer("Hello world! How are you doing today? I'm doing fantastic!")


# Now we have our tokenizer defined, we want to actually tokenize our data.
# 
# Each dataset provided by the `datasets` library is an instance of a `Dataset` class. We can see all the methods in a `Dataset` [here](https://huggingface.co/docs/datasets/package_reference/main_classes.html#dataset), but the main one we are interested in is [`map`](https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map). By using `map` we can apply a function to every example in the dataset and either update the example or create a new feature.
# 
# We define the `tokenize_example` function below which takes in an `example`, a `tokenizer` and a `max_length` argument, tokenizes the text in the example, given by `example['text']`, trims the tokens to a maximum length and then returns a dictionary with the new feature name and feature value for that example. Note that the first argument to a function which we are going to `map` must always be the example dictionary, and it must always return a dictionary where the keys are the feature names and the values are the feature values to be added to this example.
# 
# We're trimming the tokens to a maximum length here as some examples are unnecessarily long and we can predict sentiment pretty well just using the first couple of hundred tokens -- though this might not be true for you if you're using a different dataset!
# 

# In[9]:


def tokenize_example(example, tokenizer, max_length):
    tokens = tokenizer(example["text"])[:max_length]
    return {"tokens": tokens}


# We apply the `tokenize_example` function below, on both the train and test sets. Any arguments to the function -- that aren't the example -- need to be passed as the `fn_kwargs` dictionary, with the keys being the argument names and the values the value passed to that argument.
# 
# Operations on a `Dataset` are **not** performed in-place. You should always return the result into a new variable.
# 
# Note the warnings showing that as I have performed this `map` before, the results are cached and are thus loaded from the cache instead of being calculated again.
# 

# In[10]:


max_length = 256

train_data = train_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)
test_data = test_data.map(
    tokenize_example, fn_kwargs={"tokenizer": tokenizer, "max_length": max_length}
)


# We can now see that our `train_data` has a _tokens_ feature -- as "tokens" was a key in the dictionary returned by the function we used for the `map`.
# 

# In[11]:


train_data


# By looking at the `features` attribute we can see it has automatically added the information about the tokens feature -- each is a sequence (a list) of strings. A `length=-1` means that all of our token sequences are not the same length.
# 

# In[12]:


train_data.features


# We can check the first example in our train set to see the result of the tokenization:
# 

# In[13]:


train_data[0]["tokens"][:25]


# ### Creating Validation Data
# 
# Next up, we'll create a _validation set_ from our data. This is similar to our test set in that we do not train our model on it, we only evaluate our model on it.
# 
# Why have both a validation set and a test set? Your test set respresents the real world data that you'd see if you actually deployed this model. You won't be able to see what data your model will be fed once deployed, and your test set is supposed to reflect that. Every time we tune our model hyperparameters or training set-up to make it do a bit better on the test set, we are leak information from the test set into the training process. If we do this too often then we begin to overfit on the test set. Hence, we need some data which can act as a "proxy" test set which we can look at more frequently in order to evaluate how well our model actually does on unseen data -- this is the validation set.
# 
# We can split a `Dataset` using the `train_test_split` method which splits a dataset into two, creating a `DatasetDict` for each split, one called `train` and another called `test` -- a bit confusing because these are our train and validation sets, not the test. We use `test_size` to set the portion of the data used for the validation set -- 0.25 means we use 25% of the training set -- and the examples are chosen randomly.
# 

# In[14]:


test_size = 0.25

train_valid_data = train_data.train_test_split(test_size=test_size)
train_data = train_valid_data["train"]
valid_data = train_valid_data["test"]


# By showing the lengths of each split within our dataset, we can see the 25,000 training examples have now been split into 18,750 training examples and 6,250 validation examples, with the original 25,000 test examples remaining untouched.
# 

# In[15]:


len(train_data), len(valid_data), len(test_data)


# ### Creating a Vocabulary
# 
# Next, we have to build a _vocabulary_. This is look-up table where every unique token in your dataset has a corresponding _index_ (an integer).
# 
# We do this as machine learning models cannot operate on strings, only numerical vaslues. Each _index_ is used to construct a _one-hot_ vector for each token. A one-hot vector is a vector where all the elements are 0, except one, which is 1, and the dimensionality is the total number of unique tokens in your vocabulary, commonly denoted by $V$.
# 
# For example:
# 
# ![](assets/vocabulary.png)
# 
# One issue with creating a vocabulary using every single word in the dataset is that there are usually a considerable amount of unique tokens. One way to combat this is to either only construct the vocabulary only using the most commonly appearing tokens, or to only use tokens which appear a minimum amount of times in the dataset. In this notebook, we do the latter, keeping on the tokens which appear 5 times.
# 
# What happens to tokens which appear less than 5 times? We replace them with a special _unknown_ token, denoted by `<unk>`. For example, if the sentence "This film is great and I love it", but the word "love" was not in the vocabulary, it would become: "This film is great and I \<unk\> it".
# 
# We use the `build_vocab_from_iterator` function from `torchtext.vocab` to create our vocabulary, specifying the `min_freq` (the minimum amount of times a token should appear to be added to the vocabulary) and `special_tokens` (tokens which should be appended to the start of the vocabulary, even if they don't appear `min_freq` times in the dataset).
# 
# The first special token is our unknown token, the other, `<pad>` is a special token we'll use for padding sentences.
# 
# When we feed sentences into our model, we pass a _batch_ of sentences, i.e. more than one, at the same time. Passing a batch of sentences is preferred to passing sentences one at a time as it allows our model to perform computation on all sentences within a batch in paralle, thus speeding up the time taken to train and evaluate our model. All sentences within a batch need to be the same length (in terms of the number of tokens). Thus, to ensure each sentence is the same length, any shorter than the longest sentence need to have padding tokens appended to the end of them.
# 
# For an example batch of two sentences of length four and three tokens:
# 
# ![](assets/padding.png)
# 
# As we can see, the second sentence has been padded with a single `<pad>` token.
# 

# In[16]:


min_freq = 5
special_tokens = ["<unk>", "<pad>"]

vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)


# Why is the vocabulary built from only the training data? When testing any machine learning system, we want to avoid any form of data leakage but not using any information from the test data -- this includes the frequency of tokens within the test data. As the validation set is supposed to reflect the test set as much as possible, we also do not use it to build the vocabulary. A common mistake is building the vocabulary using validation and test data -- do not do this!
# 
# Now we have our vocabulary, we can first examine it by checking its length -- the number of tokens in the vocabulary.
# 

# In[17]:


len(vocab)


# We can view the tokens in our vocabulary using the `get_itos` method, which returns a list of strings (tokens), and the index of each token in the list is the index of the token in our vocabulary.
# 

# In[18]:


vocab.get_itos()[:10]


# We can get the index of a token by accessing it like a dictionary.
# 

# In[19]:


vocab["and"]


# We store the indices of the unknown and padding tokens (zero and one, respectively) in variables, as we'll use these further on in this notebook.
# 

# In[20]:


unk_index = vocab["<unk>"]
pad_index = vocab["<pad>"]


# We can check if a token is in our vocabulary using the `in` operator.
# 

# In[21]:


"some_token" in vocab


# By default, a vocabulary created by `torchtext` will throw an error if you attempt to obtain the index of a token which is not in the vocabulary, i.e. `vocab["some_token"]` will throw an error.
# 
# We need to explicity tell the vocabulary which token to return if we pass a token not in the vocabulary. We do this using the `set_default_index` method, passing in the index we wish it to return. Here, we pass the index of the unknown token.
# 

# In[22]:


vocab.set_default_index(unk_index)


# Now, when trying to get the index of a token that is not in the vocabulary, instead of throwing an error we get zero, the value of `unk_index`, our unknown token, `<unk>`.
# 

# In[23]:


vocab["some_token"]


# To look-up a list of tokens, we can use the vocabulary's `lookup_indices` method.
# 

# In[24]:


vocab.lookup_indices(["hello", "world", "some_token", "<pad>"])


# ### Numericalizing Data
# 
# Now we have our vocabulary, we can numericalize our data. This involves converting the tokens within our dataset into indices. Similar to how we tokenized our data using the `Dataset.map` method, we'll define a function that takes an example and our vocabulary, gets the index for each token in each example and then creates an `ids` field which containes the numericalized tokens.
# 

# In[25]:


def numericalize_example(example, vocab):
    ids = vocab.lookup_indices(example["tokens"])
    return {"ids": ids}


# We then apply this function to all examples in the training, validation and testing datasets.
# 

# In[26]:


train_data = train_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
valid_data = valid_data.map(numericalize_example, fn_kwargs={"vocab": vocab})
test_data = test_data.map(numericalize_example, fn_kwargs={"vocab": vocab})


# Checking an example, we can see that the `id` field now consists of the indexes of the tokens from that example.
# 

# In[27]:


train_data[0]["tokens"][:10]


# In[28]:


vocab.lookup_indices(train_data[0]["tokens"][:10])


# In[29]:


train_data[0]["ids"][:10]


# The final step of numericalization is transforming the `ids` and `label` from integers into PyTorch tensors, which we do using the `with_format` method.
# 
# We do this because our PyTorch models work with tensors, and not integers.
# 

# In[30]:


train_data = train_data.with_format(type="torch", columns=["ids", "label"])
valid_data = valid_data.with_format(type="torch", columns=["ids", "label"])
test_data = test_data.with_format(type="torch", columns=["ids", "label"])


# In[31]:


train_data[0]["label"]


# In[32]:


train_data[0]["ids"][:10]


# One thing to note is that when using `with_format`, all the columns not specified (`"tokens"` and `"text"`) are removed from the example.
# 

# In[33]:


train_data[0].keys()


# Removing the "tokens" field is fine, as if we wanted to retrieve the human-readable tokens again we can simply convert the tensor into a Python list of integers and then use the vocabulary's `lookup_tokens` method.
# 

# In[34]:


vocab.lookup_tokens(train_data[0]["ids"][:10].tolist())


# ### Creating Data Loaders
# 
# The final step of preparing the data is creating the data loaders. We can iterate over a data loader to retrieve batches of examples. This is also where we will perform any padding that is necessary.
# 
# We first need to define a function to _collate_ a batch, consisting of a list of examples, into what we want our data loader to output.
# 
# Here, our desired output from the data loader is a dictionary with keys of `"ids"` and `"label"`.
# 
# The value of `batch["ids"]` should be a tensor of shape `[batch size, length]`, where `length` is the length of the longest sentence (in terms of tokens) within the batch, and all sentences shorter than this should be padded to that length.
# 
# The value of `batch["label"]` should be a tensor of shape `[batch size]` consisting of the label for each sentence in the batch.
# 
# We define a function, `get_collate_fn`, which is passed the pad token index and returns the actual collate function. Within the actual collate function, `collate_fn`, we get a list of `"ids"` tensors for each example in the batch, and then use the `pad_sequence` function, which converts the list of tensors into the desired `[batch size, length]` shaped tensor and performs padding using the specified `pad_index`. By default, `pad_sequence` will return a `[length, batch size]` shaped tensor, but by setting `batch_first=True`, these two dimensions are switched. We get a list of `"label"` tensors and convert the list of tensors into a single `[batch size]` shaped tensor.
# 

# In[35]:


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


# Next, we define a function which returns our actual data loader. It takes in a dataset, desired batch size (the number of sentences we want in a batch), our padding token index, and if the dataset should be shuffled.
# 

# In[36]:


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader


# Finally, we get the data loaders for the training, validation and test data.
# 
# We set the batch size equal to 512. Our batch size should be set as high as we can, as larger batches means more parallel computation, less compute time, and thus faster training and evaluation.
# 
# Only the training data loader needs to be shuffled, as it's the only one used to actually tune the parameters within the model, and your training data should always be shuffled.
# 

# In[37]:


batch_size = 512

train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
test_data_loader = get_data_loader(test_data, batch_size, pad_index)


# ## Building the Model
# 

# In[38]:


class NBoW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        pooled = embedded.mean(dim=1)
        # pooled = [batch size, embedding dim]
        prediction = self.fc(pooled)
        # prediction = [batch size, output dim]
        return prediction


# In[39]:


vocab_size = len(vocab)
embedding_dim = 300
output_dim = len(train_data.unique("label"))

model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)


# In[40]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


# In[41]:


vectors = torchtext.vocab.GloVe()


# In[42]:


hello_vector = vectors.get_vecs_by_tokens("hello")


# In[43]:


hello_vector.shape


# In[44]:


hello_vector[:32]


# In[45]:


pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())


# In[46]:


pretrained_embedding.shape


# In[47]:


model.embedding.weight


# In[48]:


pretrained_embedding


# In[49]:


model.embedding.weight.data = pretrained_embedding


# In[50]:


model.embedding.weight


# In[51]:


optimizer = optim.Adam(model.parameters())


# In[52]:


criterion = nn.CrossEntropyLoss()


# In[53]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device


# In[54]:


model = model.to(device)
criterion = criterion.to(device)


# In[55]:


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


# In[56]:


def evaluate(data_loader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="evaluating..."):
            ids = batch["ids"].to(device)
            label = batch["label"].to(device)
            prediction = model(ids)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
    return np.mean(epoch_losses), np.mean(epoch_accs)


# In[57]:


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


# In[58]:


n_epochs = 10
best_valid_loss = float("inf")

metrics = collections.defaultdict(list)

for epoch in range(n_epochs):
    train_loss, train_acc = train(
        train_data_loader, model, criterion, optimizer, device
    )
    valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)
    metrics["train_losses"].append(train_loss)
    metrics["train_accs"].append(train_acc)
    metrics["valid_losses"].append(valid_loss)
    metrics["valid_accs"].append(valid_acc)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "nbow.pt")
    print(f"epoch: {epoch}")
    print(f"train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}")
    print(f"valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}")


# In[59]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_losses"], label="train loss")
ax.plot(metrics["valid_losses"], label="valid loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()


# In[60]:


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(metrics["train_accs"], label="train accuracy")
ax.plot(metrics["valid_accs"], label="valid accuracy")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_xticks(range(n_epochs))
ax.legend()
ax.grid()


# In[61]:


model.load_state_dict(torch.load("nbow.pt"))

test_loss, test_acc = evaluate(test_data_loader, model, criterion, device)


# In[62]:


print(f"test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}")


# In[63]:


def predict_sentiment(text, model, tokenizer, vocab, device):
    tokens = tokenizer(text)
    ids = vocab.lookup_indices(tokens)
    tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
    prediction = model(tensor).squeeze(dim=0)
    probability = torch.softmax(prediction, dim=-1)
    predicted_class = prediction.argmax(dim=-1).item()
    predicted_probability = probability[predicted_class].item()
    return predicted_class, predicted_probability


# In[64]:


text = "This film is terrible!"

predict_sentiment(text, model, tokenizer, vocab, device)


# In[65]:


text = "This film is great!"

predict_sentiment(text, model, tokenizer, vocab, device)


# In[66]:


text = "This film is not terrible, it's great!"

predict_sentiment(text, model, tokenizer, vocab, device)


# In[67]:


text = "This film is not great, it's terrible!"

predict_sentiment(text, model, tokenizer, vocab, device)

