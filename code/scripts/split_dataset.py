
import os
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description='split dataset path.')
    parser.add_argument('--path', default="data/pretrained_so/StackOverflow_Posts.txt", type=str, required=False, help='数据集路径')
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    with open(args.path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    split1 = int(0.8 * len(lines))
    split2 = int(0.9 * len(lines))

    with open(os.path.join(os.path.dirname(args.path), 'train.txt'), 'w', encoding='utf-8') as f:
        for line in lines[:split1]:
            f.write(line)

    with open(os.path.join(os.path.dirname(args.path), 'valid.txt'), 'w', encoding='utf-8') as f:
        for line in lines[split1:split2]:
            f.write(line)

    with open(os.path.join(os.path.dirname(args.path), 'test.txt'), 'w', encoding='utf-8') as f:
        for line in lines[split2:]:
            f.write(line)

if __name__ == "__main__":
    main()