from draw_rectangle import print_bbox


def get_tokens(txt:str):
  return txt.split()


import Levenshtein

def calculate_distance(data,findtxt):

  if type(data) == str and type(findtxt) == str and len(findtxt) > 0:
    return Levenshtein.distance(data, findtxt)
  else:
    return 1000000

import re

def get_path(url):
  expression = r'gs:\/\/(?P<bucket>.*)\/(?P<file>.*)'
  m = re.match(expression, url).groupdict()
  return m#, m["bucket"], m["file"]

test = "gs://gcs-public-data--labeled-patents/us_084.pdf"

get_path(test)

def label_file(sample, features,ocrdf):
  labels = [get_tokens(str(sample[feat])) for feat in features]
  lens = [len(f) for f in labels]
  #new_features = [ f for f,l in zip(features,lens) if l > 0]
  tokens_to_search = [token for f in labels for token in f]
  data = [(ocrdf.apply(lambda row: calculate_distance(row["text"],token), axis=1)).to_numpy() for token in tokens_to_search]
  return np.array(data), tokens_to_search, lens


def getx(features,tokens,lens,best_variation):
  all_best_tokens_index = []
  all_best_tokens_value = []
  all_best_tokens_target_token = []

  pos = 0
  for i,l in enumerate(lens):
    best_tokens_index = best_variation[i][0,:,3]
    best_tokens_value = [features[i] for _ in range(l)]
    best_tokens_target_token = tokens[pos:pos+l]

    all_best_tokens_index.extend(best_tokens_index)
    all_best_tokens_value.extend(best_tokens_value)
    all_best_tokens_target_token.extend(best_tokens_target_token)

    pos = pos + l

  return all_best_tokens_index, all_best_tokens_value, all_best_tokens_target_token


  pass


labels_file = "data/patents_dataset.xlsx"

import numpy as np
import pandas as pd
#import xlrd
from tqdm import tqdm
import itertools
import os.path

df = pd.read_excel(labels_file, sheet_name=0)
# for each file
for i in tqdm(range(df.shape[0])):
  try:
    sample = df.iloc[i]
    file_name = get_path(sample[0])["file"]
    annotation_path = "annotation/" + file_name + ".csv"
    if os.path.exists(annotation_path):
      continue

    features = sample.keys()[2:]
    ocrdf = pd.read_csv("data/" + file_name + ".csv")
    data, tokens, lens = label_file(sample, features, ocrdf)
    print(tokens)

    ocrdf["x"] = (ocrdf.loc[:, "left"] + ocrdf.loc[:, "width"] / 2) / ocrdf.loc[0, "width"]
    ocrdf["y"] = (ocrdf.loc[:, "top"] + ocrdf.loc[:, "height"] / 2) / ocrdf.loc[0, "height"]

    # consider that words in the same line are closer
    # than in different lines
    ocrdf["x"] = ocrdf["x"]/4

    positions = ocrdf.loc[:, ["x", "y"]].to_numpy()

    myData = data.T

    top_n = 4

    top_lev = np.argsort(myData, axis=0)[:top_n]
    top_lev_values = np.sort(myData, axis=0)[:top_n]

    top_postions = ocrdf.loc[top_lev.flatten(), ["x", "y"]]
    top_postions["lev"] = top_lev_values.flatten()
    top_postions["pos"] = top_lev.flatten()

    tokens_matrix = top_postions.to_numpy().reshape(top_lev.shape[0], top_lev.shape[1], 4)

    labels_best_results = []
    labels_best_results_indexes = []
    labels_best_scores = []

    pos = 0
    # for l as length of one of the labels
    # para cada label
    for l in lens:
      cluster_matrix = tokens_matrix[:, pos:pos + l, :]

      # (topn candidates, n_tokens current label, {x y lev pos})

      tokens_vars = np.transpose(cluster_matrix, axes=(1, 0, 2))

      # ( n_tokens current label,topn candidates, {x y lev pos})

      postions_scores = []
      variations = []
      for variation in itertools.product(*tokens_vars):
        # para cada combinacao de candidatos à label
        npvariation = np.array(variation)
        deviations = np.std(npvariation[:, :2], axis=0)
        deviation = np.sqrt(np.sum(np.power(deviations, 2)))  # distancia media do centro
        levenstein = np.sum(npvariation[:, 2:3], axis=0)   # distancia de levenstein média
        #score da combinacao
        score = np.exp(levenstein) * (deviation + 1)
        postions_scores.append(score)
        variations.append(npvariation)


      postions_scores = np.array(postions_scores)
      variations = np.array(variations)

      best_variations_indexes = np.argsort(postions_scores, axis=0)[:3]
      best_variations_indexes_scores = postions_scores[best_variations_indexes]
      labels_best_scores.append(best_variations_indexes_scores)
      labels_best_results_indexes.append(best_variations_indexes)
      labels_best_results.append(variations[best_variations_indexes])
      pos += l


    labels_best_results_indexes = np.array(labels_best_results_indexes)

    viable_variations = []
    viable_scores = []

    lists = [list(range(labels_best_results[0].shape[0])) for _ in range(len(labels_best_results))]

    combinations = [x for x in itertools.product(*lists)]

    print("len(combinations)", len(combinations))

    for i, variation_indexes in enumerate(combinations):

      all_labels_variation = [labels_best_results[j][k] for j, k in enumerate(variation_indexes)]
      all_labels_scores = [labels_best_scores[j][k] for j, k in enumerate(variation_indexes)]

      variation_score = np.sum(all_labels_scores)

      #join together all position for all labels of a combination
      variation_tokens = []
      for label_candidate in all_labels_variation:
        variation_tokens.extend(label_candidate[0, :, 3])

      #if no repeated tokens in more than one label
      #it is a valid option
      if np.max(np.unique(variation_tokens, return_counts=True)[1]) == 1:

        viable_variations.append(all_labels_variation)
        viable_scores.append(variation_score)

    print("number of evaluated variations",len(viable_variations))

    best_vatiation_index = np.argmin(viable_scores)

    print("best variation index", best_vatiation_index)

    best_variation = viable_variations[best_vatiation_index]
    print(best_variation)

    all_best_tokens_index, all_best_tokens_value, all_best_tokens_target_token = getx(features,tokens,lens,best_variation)


    ocrdf.at[all_best_tokens_index,"label"] = all_best_tokens_value
    ocrdf.at[all_best_tokens_index,"target"] = all_best_tokens_target_token
    ocrdf["right"] = ocrdf["left"] + ocrdf["width"]
    ocrdf["bottom"] = ocrdf["top"] + ocrdf["height"]

    tops = ocrdf.groupby(by=["label"], dropna=True)["top"].min()
    bottoms = ocrdf.groupby(by=["label"], dropna=True)["bottom"].max()
    lefts = ocrdf.groupby(by=["label"], dropna=True)["left"].min()
    rights = ocrdf.groupby(by=["label"], dropna=True)["right"].max()

    dfx = pd.merge(lefts, rights, right_index=True, left_index=True)
    dfx = pd.merge(dfx, tops, right_index=True, left_index=True)
    dfx = pd.merge(dfx, bottoms, right_index=True, left_index=True)

    print_bbox("pdf/" + file_name, dfx, "img/" + file_name + ".png")

    ocrdf.to_csv(annotation_path)
    # break # para no primeiro ficheiro
  except:
    print("error on",file_name)