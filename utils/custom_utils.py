import torch
import nltk
import spacy
spacy_en = spacy.load('en')

def create_sentence(model, sentence, input_vocab, output_vocab,
                    device, max_length=50):
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_en(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, input_vocab.init_token)
    tokens.append(input_vocab.eos_token)

    # Go through each vocab token and convert to an index
    text_to_indices = [input_vocab.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [output_vocab.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == output_vocab.vocab.stoi["<eos>"]:
            break

    generated_sentence = [output_vocab.vocab.itos[idx] for idx in outputs]
    # remove start token
    return generated_sentence[1:]

def tokenize_text(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def modified_bleu(data, model, input_vocab, output_vocab, device):
    targets = []
    outputs = []
    sum_bleu = 0
    data_input = enumerate(data)
    while len(targets) < 100:
        example = data[np.random.randint(len(data))]
        src = vars(example)["i"]
        trg = vars(example)["o"]
        prediction = create_sentence(model, src, input_vocab, output_vocab, device)
        prediction = prediction[:-1]  # remove <eos> token
        
        targets.append([trg])
        outputs.append(prediction)

    return nltk.translate.bleu_score.corpus_bleu(targets, outputs, 
           smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)

def foldify(data, k_folds):
    batches = []
    for _, batch in enumerate(data):
        batches.append(batch)
    output = []
    chunk_size = len(batches)//k_folds
    iterations = 0
    for i in range(0, len(batches), chunk_size):
        iterations += 1
        if(iterations < k_folds):
            output.append(batches[i:i+chunk_size])
        else:
            output.append(batches[i:-1])
            break
    return output