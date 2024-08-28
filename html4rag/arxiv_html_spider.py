# %%
#  retrieve htmls from arxiv
import json

import urllib3
import re
from bs4 import BeautifulSoup as bs
from tqdm import tqdm
import concurrent.futures

field = "cs"

#  max show 2000 papers
index_url = f"https://arxiv.org/list/{field}/pastweek?show=2000"
#  get all possible links under the index_url
http = urllib3.PoolManager()


def get_html_content(url):
    try:
        response = http.request('GET', url)
        #  convert bytes to string
        html = response.data.decode('utf-8')
    except Exception as e:
        print(f"Error in getting html for '{url}': {e}")
        return None
    if f"No HTML for '{paper_id}'" in html:
        # print(f"No HTML for '{paper_id}'")
        return None
    return {
        "url": url,
        "html": html,
        "qas": []
    }


if __name__ == "__main__":
    response = http.request('GET', index_url)
    soup = bs(response.data, 'html.parser')
    link_tags = soup.find_all("a")

    links = []
    # filter links with herf in the form of "https://arxiv.org/abs/2404.17343"
    links += [link['href'] for link in link_tags if
              'href' in link.attrs and re.match(r"/abs/\d{4}\.\d{5}", link['href'])]
    paper_ids = [link.split("/")[-1] for link in links]
    #  remove duplicate paper_ids
    paper_ids = list(set(paper_ids))
    print(f"len of paper_ids: {len(paper_ids)}")
    # paper_ids = paper_ids[:10]
    #  html links for paper https://arxiv.org/html/2404.15846
    html_data_lines = []

    #  clear file
    with open(f"./html_data/arxiv/arxiv_{field}.jsonl", "w") as f:
        f.write("")

    #  get html data for each paper concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for paper_id in tqdm(paper_ids):
            concat_url = f"https://arxiv.org/html/{paper_id}"
            future = executor.submit(get_html_content, concat_url)
            if future.result() is not None:
                html_data_lines.append(future.result())
                #  save html data to file
                with open(f"./html_data/arxiv/arxiv_{field}.jsonl", "a") as f:
                    f.write(json.dumps(html_data_lines[-1], ensure_ascii=False) + "\n")


    print(f"len of html_data_lines: {len(html_data_lines)}")


    # with open(f"../html_data/arxiv/arxiv_{field}.jsonl", "w") as f:
    #     for line in html_data_lines:
    #         f.write(json.dumps(line) + "\n")
