import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext.data.metrics import bleu_score
from statistics import median, variance
from functools import reduce
from operator import add

from model import Transformer
from custom_utils import create_sentence, tokenize_text, modified_bleu, foldify
from utils import save_checkpoint, load_checkpoint

input_text = Field(tokenize=tokenize_text, lower=True, init_token="<sos>", eos_token="<eos>")
output_text = Field(tokenize=tokenize_text, lower=True, init_token="<sos>", eos_token="<eos>")

fields = {'Input': ('i', input_text), 'Output': ('o', output_text)}

big_data = TabularDataset.splits(
    path="",
    train="./shuffledgutenberg.json",
    format='json',
    fields=fields
)

input_text.build_vocab(big_data[0], max_size=20_000, min_freq=8) # , vectors='fasttext.simple.300d'
output_text.build_vocab(big_data[0], max_size=20_000, min_freq=8) # , vectors='fasttext.simple.300d'

print("Input Vocab Size: {}".format(len(input_text.vocab)))
print("Output Vocab Size: {}".format(len(output_text.vocab)))

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = True
save_model = True
graph = False

# Training hyperparameters
num_epochs = 100
learning_rate = 0.0003
batch_size = 32
k_folds = 10

# Model hyperparameters
src_vocab_size = len(input_text.vocab)
trg_vocab_size = len(output_text.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.8
max_len = 100
forward_expansion = 4
src_pad_idx = input_text.vocab.stoi["<pad>"]

big_iterator = BucketIterator.splits(
    (big_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.i),
    device=device
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = input_text.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "Emma Woodhouse, handsome, clever, and rich, with a comfortable home" # Output should be: and happy disposition, seemed to unite some of the best blessings

if graph:
    plt.ion()

bleu_scores = []
big_iterator = foldify(big_iterator[0], k_folds)
big_data = foldify(big_data[0], k_folds)
k = 0
prev_score = float('-inf')
biggest_score = 0.13

for epoch in range(num_epochs):
    k += 1
    k = k % 10
    print(f"[Epoch {(epoch + 1)} / {num_epochs}]")

    if save_model and prev_score > biggest_score:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), 
        }
        save_checkpoint(checkpoint)
        biggest_score = prev_score

    model.eval()
    generated_sentence = create_sentence(
        model, sentence, input_text, output_text, device, max_length=50
    )

    print(f"Generated sentence example: \n {generated_sentence}")
    model.train()

    losses = []
    means = []
    
    temp_boi = big_iterator.copy() # I stopped caring after this point
    del temp_boi[k]
    temp_boi = reduce(add, temp_boi)
    
    for batch in temp_boi:
        # Get input and targets and get to cuda
        inp_data = batch.i.to(device)
        target = batch.o.to(device)
        # Forward prop
        output = model(inp_data, target[:-1, :])
        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        # Gradient descent step
        optimizer.step()

        losses.append(loss.item())
        means.append(median(losses[-10:]))

        if graph:
            plt.xlabel("Number of Iterations")
            plt.ylabel("Loss")
            plt.plot(losses, label="Raw Loss: {:.2f}".format(losses[-1]))
            plt.plot(means, label="Moving Average: {:.2f}".format(means[-1]))
            plt.legend(loc='upper left')
            plt.draw()
            plt.pause(0.0001)
            plt.savefig("loss_plots/Epoch-{}.png".format(epoch + 1))
            plt.clf()
    try:
        bleu_scores.append(modified_bleu(big_data[k], model, input_text, output_text, device))
        prev_score = bleu_scores[-1]
        print("Bleu Score: {:.2f}".format(prev_score))
    except Exception as e:
        print("BLEU scores failed to process")
        print(e)
    print("Loss: {:.2f}".format(losses[-1]))
    print("Averaged Loss: {:.2f}".format(means[-1]))
print("Final Score: {:.2f}".format(modified_bleu(big_data[k], 
                                                 model, 
                                                 input_text,
                                                 output_text,
                                                 device)))
print("Final Result:")
print(sentence + ' ' + ' '.join(create_sentence(
        model, sentence, input_text, output_text, device, max_length=50
    )[1:-1]))