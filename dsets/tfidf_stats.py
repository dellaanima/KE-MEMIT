import json
from itertools import chain
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from dsets import AttributeSnippets
from util.globals import *

REMOTE_IDF_URL = f"{REMOTE_ROOT_URL}/data/dsets/idf.npy"
REMOTE_VOCAB_URL = f"{REMOTE_ROOT_URL}/data/dsets/tfidf_vocab.json"

from tqdm import tqdm  # 프로그레스 바를 위한 라이브러리

def get_tfidf_vectorizer(data_dir: str):
    data_dir = Path(data_dir)

    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"
    if not (idf_loc.exists() and vocab_loc.exists()):
        collect_stats(data_dir)

    idf = np.load(idf_loc)
    vocab = {}

    try:
        with open(vocab_loc, "r") as f:
            # 파일의 총 라인 수를 파악하여 프로그레스 바의 최대 값으로 설정
            total_lines = sum(1 for line in f)
            f.seek(0)  # 파일 포인터를 다시 파일 시작으로 이동

            # 파일을 줄 단위로 읽어서 처리하여 문제가 있는 부분을 건너뛸 수 있습니다.
            json_content = ''
            with tqdm(total=total_lines, desc="Reading TF-IDF vocabulary") as pbar:
                for line in f:
                    try:
                        json_content += line
                        # 중간 검증을 통해 JSON 형식이 올바른지 확인
                        json.loads(json_content)
                        pbar.update(1)  # 프로그레스 바 업데이트
                    except json.JSONDecodeError:
                        pbar.update(1)  # 오류가 있어도 프로그레스 바 업데이트
                        continue  # 오류가 발생하면 다음 줄로 넘어갑니다.

            # 마지막으로 검증된 JSON 내용을 사용
            vocab = json.loads(json_content)
    except Exception as e:
        print(f"Error while reading {vocab_loc}: {e}")

    class MyVectorizer(TfidfVectorizer):
        TfidfVectorizer.idf_ = idf

    vec = MyVectorizer()
    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))

    return vec


#def get_tfidf_vectorizer(data_dir: str):
#    """
#    Returns an sklearn TF-IDF vectorizer. See their website for docs.
#    Loading hack inspired by some online blog post lol.
#    """
#
#    data_dir = Path(data_dir)
#
#    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"
#    if not (idf_loc.exists() and vocab_loc.exists()):
#        collect_stats(data_dir)
#
#    idf = np.load(idf_loc)
#    with open(vocab_loc, "r") as f:
#        vocab = json.load(f)
#
#    class MyVectorizer(TfidfVectorizer):
#        TfidfVectorizer.idf_ = idf
#
#    vec = MyVectorizer()
#    vec.vocabulary_ = vocab
#    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))
#
#    return vec





def collect_stats(data_dir: str):
    """
    Uses wikipedia snippets to collect statistics over a corpus of English text.
    Retrieved later when computing TF-IDF vectors.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"

    try:
        print(f"Downloading IDF cache from {REMOTE_IDF_URL}")
        torch.hub.download_url_to_file(REMOTE_IDF_URL, idf_loc)
        print(f"Downloading TF-IDF vocab cache from {REMOTE_VOCAB_URL}")
        torch.hub.download_url_to_file(REMOTE_VOCAB_URL, vocab_loc)
        return
    except Exception as e:
        print(f"Error downloading file:", e)
        print("Recomputing TF-IDF stats...")

    snips_list = AttributeSnippets(data_dir).snippets_list
    documents = list(chain(*[[y["text"] for y in x["samples"]] for x in snips_list]))

    vec = TfidfVectorizer()
    vec.fit(documents)

    idfs = vec.idf_
    vocab = vec.vocabulary_

    np.save(data_dir / "idf.npy", idfs)
    with open(data_dir / "tfidf_vocab.json", "w") as f:
        json.dump(vocab, f, indent=1)
