"""
script to replace premise and hypothesis in snli data with backtranslations from a txt files.
"""

import pandas as pd
import argparse

def backtranslate_snli(df_path, s1_path, s2_path, out_dir, fname, asym):
    # Read in the original SNLI dataset as a pandas dataframe
    df = pd.read_csv(df_path, sep='\t', header=0)

    # Read in the back-translated premise and hypothesis from the provided text files as lists of strings
    with open(s1_path) as f1:
        bt_s1 = f1.read().splitlines()
    with open(s2_path) as f2:
        bt_s2 = f2.read().splitlines()

    # Make sure that the number of back-translated sentences matches the number of rows in the original dataset
    assert len(df) == len(bt_s2)

    # Replace either the premise or hypothesis in the original dataset with the corresponding back-translated sentences
    if asym == "s1":
        df["sent1_bt"] = bt_s1
        df.drop("sentence1", axis=1, inplace=True)
        df.columns = ["sentence1" if x == "sent1_bt" else x for x in df.columns]
    else:
        df["sent2_bt"] = bt_s2
        df.drop("sentence2", axis=1, inplace=True)
        df.columns = ["sentence2" if x == "sent2_bt" else x for x in df.columns]

    # Save the modified dataset to the specified output directory
    df.to_csv(f"{out_dir}{fname}.txt", sep="\t", index=False)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Back-translate SNLI dataset")
    parser.add_argument("df", help="Path to the original SNLI dataset")
    parser.add_argument("s1", help="Path to the back-translated premise text file")
    parser.add_argument("s2", help="Path to the back-translated hypothesis text file")
    parser.add_argument("out", help="Directory where the back-translated dataset will be saved")
    parser.add_argument("fname", help="Name of the back-translated dataset file, without the file extension")
    parser.add_argument("asym", help="Indicates whether the premise or hypothesis will be back-translated (s1 or s2)")
    args = parser.parse_args()

    backtranslate_snli(args.df, args.s1, args.s2, args.out, args.fname, args.asym)

