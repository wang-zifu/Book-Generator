import nltk
from nltk.corpus import gutenberg as gut
from nltk.tokenize import word_tokenize
import pandas as pd
import multiprocessing as mp
from sklearn.model_selection import train_test_split
import os

nltk.download('gutenberg')

TRAIN_PATH = '../nltk_data/nltktrain/'
TEST_PATH = '../nltk_data/nltktest/'
VALIDATION_PATH = '../nltk_data/nltkvalidation/'

def convert_to_json_split(filename):
    try:
        input_txt = gut.raw(filename).split('\n')
        input_txt = [line for line in input_txt if line != ""]
        output_txt = input_txt[1:]
        raw_data = {'Input': input_txt[:-1], 
                    'Output': output_txt}
        df = pd.DataFrame(raw_data, columns=['Input', 'Output'])

        train, test = train_test_split(df, test_size=0.25)
        valid, test = train_test_split(test, test_size=0.4)

        train.to_json(os.path.join(TRAIN_PATH,'train-{}.json'.format(filename)), orient='records', lines=True)
        test.to_json(os.path.join(TEST_PATH, 'test-{}.json'.format(filename)), orient='records', lines=True)
        valid.to_json(os.path.join(VALIDATION_PATH, 'validation-{}.json'.format(filename)), orient='records', lines=True)

        print("Processed {}".format(filename))
        return df
    except Exception as e:
        print('Error {} occurred'.format(e))
        print('Failed to process {}'.format(filename))

if __name__ == "__main__":
    files = gut.fileids()
    frame = []
    with mp.Pool() as pool:
        frame = pool.map(convert_to_json_split, files)
    df = pd.concat(frame)
    train, test = train_test_split(df, test_size=0.25)
    valid, test = train_test_split(test, test_size=0.4)
    train.to_json('guttrain.json', orient='records', lines=True)
    test.to_json('guttest.json', orient='records', lines=True)
    valid.to_json('gutvalid.json', orient='records', lines=True)
    print("Finished!!!")

    
