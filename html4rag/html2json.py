from bs4 import BeautifulSoup as bs
import os
from datetime import datetime
from tqdm import tqdm
from collections import Counter
import re
import argparse
import json
from readability import Document

domin = "ruc"
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="", type=str, help='directory containing html pages')
parser.add_argument('--output_dir', type=str, default='', help='directory containing documents')
parser.add_argument('--url_file', default="", type=str, help='converting html page name to url, checking the source type')
args = parser.parse_args()

def find_date(list_of_strings):
    date_pattern = r"\d{4}.\d{2}.\d{2}"
    for string in list_of_strings:
        if re.search(date_pattern, string):
            l = re.search(date_pattern, string).group()
            l = [l[:4], l[5:7], l[8:10]]
            date = '-'.join(l)
            return date
    return None


def process_old(input_dir, code2url_list=None):
    files = os.listdir(input_dir)
    files2url = {}
    with open(code2url_list) as f:
        for line in f:
            line = line.strip().split('\t')
            files2url[line[0]] = line[1]

    all_files_processed = []
    sentences_freq = []
    for file in tqdm(files):
        code = file.split('.')[0]
        url = files2url[code]
        if url.endswith('pdf') or url.endswith('doc') or url.endswith('docx') or url.endswith('zip'):
            continue
        soup = bs(open(os.path.join(input_dir, file),encoding='utf-8'), 'html.parser')
        texts = soup.find_all(text=True)
        texts = [((i.parent.name if i.parent.name else "") , i.text.strip()) for i in texts if i.text.strip()]
        title = ""
        for i in texts:
            if i[0] in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                title = i[1]
                break
        if title == "":
            for i in texts:
                if i[0]  == 'title':
                    title = i[1]
                    break
        texts = [i[1] for i in texts]
        date = find_date(texts)
        if len(texts) > 0:
            all_files_processed.append([title, url, texts, date, code])
        # else: # remove annotation to split documents to passages
        #     for i in range(0, len(texts), 100):
        #         all_files_processed.append([title, url, texts[i:i+100]])
        sentences_freq += texts
    sentences_freq = Counter(sentences_freq)
    all_files_processed = [{'title': i[0], 'url': i[1], 'date': i[3], 'contents': '\n'.join(i[2]), 'id': i[4]} for i in all_files_processed]
    return all_files_processed

def clean(html, domin = "ruc_rdzs"):
    if domin == "ruc_rdzs":
        soup = bs(html, 'html.parser')
        # print(file)
        clean_tags = soup.select(".y_tit_box, .container.content")
        soup = bs("".join([str(tag) for tag in clean_tags]), 'html.parser')
        
    else:
        if domin == "ruc":
            html = html.read()
        doc = Document(html, min_text_length=6)
        summary = doc.summary()
        soup = bs(summary, 'html.parser')
    
    clean_texts = soup.find_all(text=True)
    # print(clean_texts)
    clean_texts = [i.text.strip() for i in clean_texts if i.text.strip()]
    return clean_texts

def input_html(input_path, domin):
    if domin == "finance":
        count = 0
        for file in os.listdir(input_path):
            with open(os.path.join(input_path, file)) as f:
                for line in f:
                    item = json.loads(line)
                    # print(item.keys())
                    # code = item["__ts__"]
                    html = item["html"]
                    url = item["url"]
                    yield count, url, html
                    count += 1


def process(input_dir, code2url_list=None):
    files = os.listdir(input_dir)
    files2url = {}
    with open(code2url_list) as f:
        for line in f:
            line = line.strip().split('\t')
            files2url[line[0]] = line[1]

    all_files_processed = []
    sentences_freq = []

    for id in tqdm(range(len(files))):
        code = str(id)
        file = code + ".html"
        code = file.split('.')[0]
        url = files2url[code]
        if url.endswith('pdf') or url.endswith('doc') or url.endswith('docx') or url.endswith('zip'):
            continue
        soup = bs(open(os.path.join(input_dir, file),encoding='utf-8'), 'html.parser')
        # print(file)
        # clean_tags = soup.select(".y_tit_box, .container.content")
        # soup2 = bs("".join([str(tag) for tag in clean_tags]), 'html.parser')
        # clean_texts = soup2.find_all(text=True)
        # clean_texts = [i.text.strip() for i in clean_texts if i.text.strip()]

        clean_texts = clean(open(os.path.join(input_dir, file),encoding='utf-8'), domin)
        # print(file)
        # print(clean_texts)

        # print(clean_texts)
        texts = soup.find_all(text=True)

        texts = [((i.parent.name if i.parent.name else "") , i.text.strip()) for i in texts if i.text.strip()]
        title = ""
        for i in texts:
            if i[0] in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                title = i[1]
                break
        if title == "":
            for i in texts:
                if i[0]  == 'title':
                    title = i[1]
                    break

        texts = [i[1] for i in texts]
        date = find_date(texts)
        if len(texts) > 0:
            all_files_processed.append([title, url, clean_texts, date, code])
        
        sentences_freq += clean_texts


    sentences_freq = Counter(sentences_freq)
    all_files_processed = [{'title': i[0], 'url': i[1], 'date': i[3], 'contents': '\n'.join(i[2]), 'id': i[4]} for i in all_files_processed]
    return all_files_processed

if args.url_file == '':
    processed = process(args.input_dir)
else:
    processed = process(args.input_dir, args.url_file)

hashes = set()
new_processed = []
for i in processed:
    if len(''.join(i['contents'])) < 100000 and hash(i['contents']) not in hashes:
        hashes.add(hash(i['contents']))
        new_processed.append(i)
    else:
        pass

for d in new_processed:
    file_name = str(d['id']) + '.json'
    if domin == "ruc":
        d["id"] = int(d["id"]) + 3155
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, file_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(d, ensure_ascii=False))