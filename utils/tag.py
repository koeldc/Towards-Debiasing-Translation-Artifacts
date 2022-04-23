import argparse

punc = '''!"$%'()*+,-./:;=?@[]_`{¡£§«°²´·¸»½¾¿–—‘’‚“”„…€''' #'''!()-[]{};:'"\,<>./?@#$%^&*_~''' # """!"%'()+,-./:;?@[]¡«°´·»¿–‘’“”…€"""

puncs = set()

for ch in punc:
    puncs.add(ch)

#print(puncs)

def suffix_words(wordlist, suf):
    "fn. to add suffix to words in snli datasets"
    #print("before:", wordlist)
    new_list = []

    #uniq_wds = set(text.split())
    #print(uniq_wds)
    
    for wd in wordlist:
        word = wd+f"_{suf}" 
        new_list.append(word)
    #print("after", new_list)

    return new_list


def suffix_puncts(args):
    with open(args.f) as f:
        data = f.read()
        uniq_wds = set(data.split())
        #print(uniq_wds)
        #print(f"num uniq words: {len(uniq_wds)}")
        for wd in uniq_wds:
            if wd in puncs:
                data = data.replace(wd, wd+f"_{args.suf}")
    
    with open(args.dir+args.fname, "w") as of:
        of.write(data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to file that needs to be suffixed", default="mono/txt_split/es_de/es.txt")
    parser.add_argument("--suffix", help="suffix to add", default="og")
    parser.add_argument("--dir", help="output dir to save the tagged file", default="mono/txt_split/es_de/")
    parser.add_argument("--fname", help="output filename", default="es_tagged.txt")

    args = parser.parse_args()

    suffix_puncts(args)

