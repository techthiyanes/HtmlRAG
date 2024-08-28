import argparse
import json
from multiprocessing import Process

import loguru
from bs4 import BeautifulSoup
from bs4.element import Comment
from tqdm import tqdm


def simplify_html(soup):
    for script in soup(["script", "style"]):
        script.decompose()
    #  remove all attributes
    for tag in soup.find_all(True):
        tag.attrs = {}
    #  remove empty tags recursively
    while True:
        removed = False
        for tag in soup.find_all():
            if not tag.text.strip():
                tag.decompose()
                removed = True
        if not removed:
            break
    #  remove href attributes
    for tag in soup.find_all("a"):
        del tag["href"]
    #  remove comments
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        comment.extract()

    def concat_text(text):
        text = "".join(text.split("\n"))
        text = "".join(text.split("\t"))
        text = "".join(text.split(" "))
        return text

    # remove all tags with no text
    for tag in soup.find_all():
        children = [child for child in tag.contents if not isinstance(child, str)]
        if len(children) == 1:
            tag_text = tag.get_text()
            child_text = "".join([child.get_text() for child in tag.contents if not isinstance(child, str)])
            if concat_text(child_text) == concat_text(tag_text):
                tag.replace_with_children()
    #  if html is not wrapped in a html tag, wrap it

    # remove empty lines
    res = str(soup)
    lines = [line for line in res.split("\n") if line.strip()]
    res = "\n".join(lines)
    return res


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--rewrite_method", type=str, default="t5")
    argparser.add_argument("--split", type=str, default="test")
    args = argparser.parse_args()
    rewrite_method = args.rewrite_method
    split = args.split

    # datasets = ["asqa", "hotpot-qa", "nq", "trivia-qa", "musique"]
    datasets = ["trivia-qa"]

    def simplify_html_lines(dataset):
        data_file = f"./html_data/{dataset}/bing/binghtml-{rewrite_method}-{dataset}-{split}.jsonl"
        data_lines = [json.loads(l) for l in open(data_file)]
        output_file = f"./html_data/{dataset}/bing/binghtml-{rewrite_method}-{dataset}-simple-{split}.jsonl"
        loguru.logger.info(f"Reading data from {data_file}")
        for idx in tqdm(range(len(data_lines)), desc=f"Processing {dataset} {split}"):
            for idj in range(len(data_lines[idx][f'{rewrite_method}_results'])):
                h_soup = BeautifulSoup(data_lines[idx][f'{rewrite_method}_results'][idj]['html'], 'html.parser')
                data_lines[idx][f'{rewrite_method}_results'][idj]['html'] = simplify_html(h_soup)

        with open(output_file, "w") as f:
            #  try to encode in utf-8, if it fails, try to replace the characters with ?
            for idl, l in enumerate(data_lines):
                try:
                    f.write(json.dumps(l, ensure_ascii=False) + "\n")
                except Exception as e:
                    loguru.logger.error(f"Line {idl} Error: {e}")
                    f.write(json.dumps(l, ensure_ascii=False).encode("utf-8", errors="replace").decode("utf-8") + "\n")
        loguru.logger.info(f"Saved simplified html to {output_file}")


    child_processes = []
    for dataset in datasets:
        p = Process(target=simplify_html_lines, args=(dataset,))
        child_processes.append(p)
        p.start()

    for p in child_processes:
        p.join()
    loguru.logger.info("All done")
