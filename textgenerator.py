import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")

def predict(payload):
    input_length = len(payload)
    tokens = tokenizer.encode(payload, return_tensors="pt").to("cuda")
    prediction = model.generate(tokens, max_length=30, do_sample=True)
    return tokenizer.decode(prediction[0])

def write(string, filename):
    with open(os.path.join('./output/', filename),"a") as output:
        output.write('\n{}'.format(string))

seed = "This book was made by an AI."
filename = "GPT2 Output.txt"

# write('Max Length: 30', filename)

curr_len = len(seed.split(' '))
total_words = curr_len

write(seed, filename)

first_time = True

while total_words < 90000:
    torch.cuda.empty_cache()
    temp_pred = predict(prediction if not first_time else seed)
    while len(tokenizer.encode(temp_pred, return_tensors="pt").to("cuda")) == 0:
        temp_pred = predict(prediction if not first_time else seed)
    prediction = temp_pred
    pred_arr = prediction.split(' ')
    prediction = ' '.join(pred_arr[curr_len:])
    write(prediction, filename)
    curr_len = len(pred_arr) - curr_len
    total_words += curr_len

    print(prediction)
    print(total_words)
    if first_time:
        first_time = False

print(total_words)