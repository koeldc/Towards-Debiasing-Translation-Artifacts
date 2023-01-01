import os
import re
from pathlib import Path
import argparse
import fasttext

def preprocess(text: str) -> str:
    """
    : param text: The text input with punctuation
    : return: unpunctuated string
    """
    pp = re.sub(r'[^\w\s]',' ', text)
    pp = re.sub(' +', ' ', pp) # remove extra spaces
    return pp

def train_fasttext_embeddings(file: str) -> fasttext.FastText:
    model = fasttext.train_unsupervised(file, min_count=5, dim=300)
    return model

def save_embeddings(model: fasttext.FastText, out_path: str):
    words = model.get_words()
    with open(out_path, "w") as fout:
        # the first line contains number of total words and vector dimension
        fout.write(f"{len(words)} {model.get_dimension()}\n")
        # line by line, append vectors to VEC file
        for w in words:
            v = model.get_word_vector(w)
            vstr = " ".join([str(vi) for vi in v])
            fout.write(f"{w} {vstr}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", help="path to the directory containing txt files", default="mono/tokenized/")
    parser.add_argument("--outdir", help="directory that would contain embeddings", default="mono/embeddings/fasttext/")
    parser.add_argument("--pattern", help="pattern to grab files from folder", default="en_tagged.txt")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Generate embeddings for each matching file in indir
    for path in Path(args.indir).rglob(args.pattern):
        model = train_fasttext_embeddings(str(path))
        save_embeddings(model, args.outdir + path.name)

