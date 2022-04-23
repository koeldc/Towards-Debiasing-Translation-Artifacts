"""
script to replace premise and hypothesis in snli data with backtranslations from a txt files.
"""

import pandas as pd
from pathlib import Path
import argparse


def main(args):

    Path(args.out).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.df, sep='\t', header=0)
    print(df.columns.values)
    #del df['label5']
    #df.dropna(axis=1, inplace=True) # drop columns with missing values


    with open(args.s1) as f1:
        bt_s1 = f1.readlines()
        bt_s1 = [s.rstrip() for s in bt_s1]
        print(len(bt_s1))
        print(bt_s1[:5])


    with open(args.s2) as f2:
        bt_s2 = f2.readlines()
        bt_s2 = [s.rstrip() for s in bt_s2]
        print(len(bt_s2))
        print(bt_s2[:5])

    assert(len(df) == len(bt_s2))

    #old_sent1 = df["sentence1"]
    #old_sent2 = df["sentence2"]

    if args.asym == "s1":
        df["sent1_bt"] = bt_s1
        df.drop('sentence1', axis = 1, inplace=True)
        df.columns = ['sentence1' if x=='sent1_bt' else x for x in df.columns]
    else:
        df["sent2_bt"] = bt_s2
        df.drop('sentence2', axis = 1, inplace=True)
        df.columns = ['sentence2' if x=='sent2_bt' else x for x in df.columns]

    #print(all(df["sent2_bt"].values == df["sentence2"].values))
    #print("before\t", df['sentence2'].values[:5])
    
    # df = df.drop(['sentence1', 'sentence2'], axis = 1)
    # df.columns = ['sentence1' if x=='sen1_bt' else 'sentence2' if x=='sent2_bt' else x for x in df.columns]
    print(df.columns.values)
    # df.drop('sentence2', axis = 1, inplace=True)
    # df.columns = ['sentence2' if x=='sent2_bt' else x for x in df.columns]

    # print(df.columns.values)
    #print("after\t", df['sentence2'].values[:5])
    #print(df['sentence1'].head(), df['sentence2'].head(), df['gold_label'].head())
    df = df[:54958] # for 10% data
    df.to_csv(args.out+args.fname+".txt", sep='\t', index=False)
    # print(len(df))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='back-translate snli dataset')
    parser.add_argument("--df", default="/home/chowdhury/snli/snli_1.0/snli_1.0_train.txt")
    parser.add_argument("--s1", default="/home/chowdhury/bt_data/train.sent1.en.bt")
    parser.add_argument("--s2", default="/home/chowdhury/bt_data/train.sent2.en.bt")
    parser.add_argument("--out", default="bt-premise-snli/")
    parser.add_argument("--fname", default="snli_train", help="without extension")
    parser.add_argument("--asym", default="s2", help="if asym=s2, hypothesis is back-translated and out = bt-hypo-snli/")
    args = parser.parse_args()
    #main(args)
    df = pd.read_csv(args.out+args.fname+".txt", sep='\t', header=0)
    df = df[:54958] # for 10% data
    print(df.gold_label.value_counts())
    df.to_csv(args.out+args.fname+"_54k.txt", sep='\t', index=False)

