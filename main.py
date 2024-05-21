import pandas as pd
import streamlit as st
import numpy as np

st.write("# Analysis for Nathan")

small = pd.read_csv("4k_all.csv")
big = pd.read_csv("32k_all.csv")

#ranking = ["yarn", "ntk", "pi", "llama"]
# best is right
ranking = np.array(["llama", "yarn", "ntk", "pi"])

data = small

# print some words
words = [str(x) for x in data["ground_truth_token"].tolist()]

yarn = data["softmax_score_yarn"].tolist()
ntk = data["softmax_score_ntk"].tolist()
pi = data["softmax_score_pi"].tolist()
llama = data["softmax_score_llama2-hf"].tolist()

def compare(i, ranking):
    y = yarn[i]
    n = ntk[i]
    p = pi[i]
    l = llama[i]
    names = np.array(["yarn", "ntk", "pi", "llama"])
    idxs = np.argsort([y,n,p,l])
    this_ranking = names[idxs]
    return (this_ranking == ranking).all()

N = 100
st.write(" ".join(
    f":red[{w}]" if compare(i, ranking) else w
    for i,w in enumerate(words[:N])
))

