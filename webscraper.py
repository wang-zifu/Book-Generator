from urllib import request
import re

gutenberg_template = 'https://www.gutenberg.org/ebooks/1.txt.utf-8'

NUM_OF_BOOKS = 10


def download_txt_file(url, number):
    response = request.urlopen(url)
    text = response.read()
    text_str = str(text)
    lines = text_str.split("\\n")
    dest_url = r"dataset\{}.txt".format(number)
    fx = open(dest_url, "w")
    for line in lines:
        re.search(r'\bis\b', "Language: English")
    for line in lines:
        fx.write(line + "\n")
    fx.close()


download_txt_file(gutenberg_template, 1)

# for i in range(1, NUM_OF_BOOKS):
#     url = "https: // www.gutenberg.org/ebooks/{}.txt.utf-8".format(i)
#     download_txt_file(url, i)
