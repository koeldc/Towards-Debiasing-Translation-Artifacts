import re

def suffix_words(wordlist, suf):
    "fn. to add suffix to words in snli datasets"
    return [wd+f"_{suf}" for wd in wordlist]

def suffix_puncts(f, suf, out_dir, out_fname):
    with open(f) as infile, open(f"{out_dir}/{out_fname}", "w") as outfile:
        for line in infile:
            outfile.write(re.sub(r"\b([\!\"\$\%\'\(\)\*\+\,\-\.\/\:\;\=\?\@\[\]\_\`\{\}\¡\£\§\«\°\²\´\·\¸\»\½\¾\¿\–\—\‘\’\‚\“\”\„\…\€])\b", r"\1_og", line))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file that needs to be suffixed", default="mono/txt_split/es_de/es.txt")
    parser.add_argument("--suffix", help="suffix to add", default="og")
    parser.add_argument("--dir", help="output dir to save the tagged file", default="mono/txt_split/es_de/")
    parser.add_argument("--fname", help="output filename", default="es_tagged.txt")
    args = parser.parse_args()
    suffix_puncts(args.file, args.suffix, args.dir, args.fname)

