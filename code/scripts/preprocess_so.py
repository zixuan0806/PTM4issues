import re
import html
import tqdm


def convert(in_path, out_path):
    title_pattern = re.compile('Title="(.*?)"')
    body_pattern = re.compile('Body="(.*?)"')
    p_pattern = re.compile('<p>(.*?)</p>')
    with open(out_path, 'w', encoding='utf-8') as fout:
        with open(in_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f):
                line_data = ""
                title_obj = title_pattern.search(line)
                if title_obj is not None:
                    line_data += title_obj.group(1)
                    fout.write(title_obj.group(1) + '\n')
                body_obj = body_pattern.search(line)
                if body_obj is not None:
                    line_data += " " + body_obj.group(1)
                line_data = html.unescape(line_data)
                # print(line_data)
                for p in p_pattern.finditer(line_data):
                    fout.write(p.group(1) + '\n')


def main():
    convert('data/pretrained_so/StackOverflow_Posts.xml', 'data/pretrained_so/StackOverflow_Posts.txt')


if __name__ == "__main__":
    main()
