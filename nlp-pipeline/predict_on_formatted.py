#!/usr/bin/env python
# coding: utf-8
import pandas as pd
#import json
#import yaml
import argparse
import os
import numpy as np
import torch
from functools import partial
import segm_prep_utils as spu
from segment_allennlp import (PosDatasetReader, #LstmTagger,
                              #AdvancedBucketIterator, permute_token, 
                              AdvancedSentenceTaggerPredictor)
#from sklearn import metrics as mtx
from mleval import get_summary

############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str,
    help='file path of the JSONL-formatted labelled documents exported from Doccano')

group = parser.add_mutually_exclusive_group(required=True)

parser.add_argument('-d', '--destination', help='output directory',
                    default='data/result', type=str)

group.add_argument('-o', '--output', help='output',
                    type=str)

group.add_argument('-t', '--tag', help='model tag',
                    default = None, type=str)

parser.add_argument('-m', '--model', help='pytorch model',
                    type=str)

args = parser.parse_args()

if args.output:
    destination_base = args.outpuit.replace('.csv', '') 
else:
    destination_base = os.path.join( args.destination,
        os.path.basename(args.input).replace('.csv','') + '-' + args.tag)

fn_prob = destination_base + '.csv'
fn_html = destination_base + '.html'
############################################################################
print('SAVING TO:', fn_prob)

reader = PosDatasetReader(index_sep='|', word_tag_delimiter='#%#')
dataset_ = reader.read(args.input)

tokenizer = spu.get_spacy_tokenizer(special_cases_file='special-cases-spacy.yaml')
model = torch.load(args.model)
predictor = AdvancedSentenceTaggerPredictor(model, dataset_reader=reader,
                                           tokenizer=tokenizer)

############################################################################
df_result = pd.DataFrame(list(map(predictor.predict_on_sentence, dataset_)))
df_result.index = [x.index for x in dataset_]
df_result.index.name='title'
df_tokens = pd.DataFrame.from_dict({kk: df_result[kk].apply(lambda x:pd.Series(x)).stack() \
                                        for kk in ['token','label']})
df_logits = predictor.logit_list_to_table(df_result['logit'])

df_prob = df_logits.applymap(lambda x: np.exp(x)).div(
    df_logits.applymap(lambda x: pd.np.exp(x)).sum(1),
    axis=0)

df_groundtruth = PosDatasetReader.long_table(dataset_)
df_groundtruth_onehot = df_groundtruth.apply(lambda x: pd.Series({y:x==y for y in df_logits.columns}))

df_prob = df_prob.loc[df_groundtruth_onehot.index, df_groundtruth_onehot.columns]

df_prob.to_csv(fn_prob)

############################################################################
summary = get_summary(df_groundtruth_onehot, df_prob)
print(pd.Series(summary))
############################################################################
print('HTML:', fn_html)

def get_start_end(gg):
    length = gg['token'].map(len) + 1 # add space
    breaks = np.cumsum([0] + length.tolist())
    gg['start'] = breaks[:-1]
    gg['end'] = breaks[1:]
    return gg

df_tokens = df_tokens.groupby('title').apply(get_start_end)
ds_texts = df_tokens.groupby('title').apply(lambda row: ' '.join(row['token'].tolist()) )
ds_texts.name = 'text'
ds_seq_ann = spu.token_table_to_dictionaries(df_tokens)
df_segments = pd.merge(ds_texts, ds_seq_ann, left_index=True, right_index=True)
ds_html = df_segments.reset_index().apply(partial(spu.vis_segments_html, title='title'), 1)


# def res_to_html(row):
#     ann_text = ' '.join([spu.get_classy_html(to, la) for to,la in zip(row['token'], row['label'])])
#     return '<div class="title">' + row['title'] + '</div>&emsp;&emsp;<div class="text">' + ann_text + '</div><br/>\n\n'
# 
# ds_html = df_result.reset_index().apply(res_to_html, axis=1)

with open(fn_html, 'w') as fh:
     header = '<link rel="stylesheet" type="text/css" href="custom.css"/>\n<body>\n\n'
     footer = '</body>'
     fh.write(header)
     for _, line in ds_html.items():
         fh.write('<p>' + line + '</p><br/>\n')
     fh.write(footer)

############################################################################


