import os, re
from pathlib import Path
import pandas as pd
import fasttext
import argparse
from time import sleep


def preprocess(text) -> str:
    """
    : param text: The text input with punctuation
    : return: unpunctuated string
    """
    pp = re.sub(r'[^\w\s]',' ',text)
    pp = re.sub(' +', ' ', pp) # remove extra spaces
    return pp


def train_fasttext_embeddings(file):
    model = fasttext.train_unsupervised(file, minCount=5, dim=300)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # (description='')
    parser.add_argument("--indir", help="path to the directory containing txt files",
                        default="mono/tokenized/")
    parser.add_argument("--outdir", help="directory that would contain embeddings",
                        default="mono/embeddings/fasttext/")
    parser.add_argument("--pattern", help="pattern to grab files from folder", default="en_tagged.txt")
    args = parser.parse_args()
    pathlist = Path(args.indir).rglob(args.pattern)

    for path in pathlist:
        path = str(path)
        print(path)
        #folder = path[path.rfind("_")-2:path.rfind("/")+1]
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        #
        # # preprocess the "en" translation and save to file
        # df["preprocessed_en"] = df.apply(lambda row: preprocess(row['text']), axis=1)
        # df.to_csv(outdir+"/train.txt", columns=["preprocessed_en"], header = False, index = False)

        # generate embeddings
        model = train_fasttext_embeddings(path)
        words = model.get_words()

        print(words)

        # save embeddings
        #emb_outdir = "fasttext_embeddings/" + file[file.find("/") + 1:file.rfind("/")]
        #Path(emb_outdir).mkdir(parents=True, exist_ok=True)
        # save the model as bin file
        #model.save_model(emb_outdir+"model.bin")

        with open(args.outdir+path[path.rfind("/")+1:], "w") as fout:
            # the first line contains number of total words and vector dimension
            fout.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")

            # line by line, append vectors to VEC file
            for w in words:
                v = model.get_word_vector(w)
                vstr = " ".join([str(vi) for vi in v])
                try:
                    fout.write(w + " " + vstr + '\n')
                except:
                    pass
  
