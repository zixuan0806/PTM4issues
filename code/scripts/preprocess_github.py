import re
import html
import tqdm
import json
import glob
import os


def convert(in_path, out_path):
    files = glob.glob(os.path.join(in_path, "*/*.txt"), recursive=True)
    with open(out_path, 'w', encoding="utf-8") as fout:
        for each_file in tqdm.tqdm(files):
            print(each_file)
            with open(each_file, 'r', encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)

                    text = obj['title']
                    if len(text.strip()) > 0:
                        fout.write(text + '\n')

                    text = obj['description']
                    if len(text.strip()) > 0:
                        fout.write(text + '\n')


def main():
    convert('data/data_view100', 'data/pretrained_github/GitHub_issues.txt')

if __name__ == "__main__":
    main()
