#!/usr/bin/env python
# coding: utf-8
#from typing import Iterator, List, Dict
import torch
torch.manual_seed(1)

import os
import json
import argparse
import numpy as np
import pandas as pd
#import re
import regex
#from functools import reduce
from functools import partial
#from copy import deepcopy
import segm_prep_utils as spu
import segment_allennlp as sgm
import dask
import dask.dataframe as dd
import dask.array as da
import multiprocessing

# def merge_annotation2json(tokens, sentences):
#     token_positions = align_token_sentences_df(tokens, sentences)
#     dfres_comb = pd.concat([token_positions, df_tokens[['labels']]], axis=1, join='inner')
# 
#     df_segments = dfres_comb.groupby(level=0).apply(partial(get_predicted_segments, label_col='labels'))
#     df_segments = pd.Series({kk:vv.to_dict(orient='records') for kk,vv in df_segments.groupby('title')})
#     # df_segments.name = 'seq_annotations'
#     df_segments = pd.concat({'seq_annotations':df_segments,
#                                   'text':sentences}, 
#                                   axis=1, join='inner')
#     df_segments['id'] = df_segments.index
#     return df_segments


# def rename_label(anns, labels={'separator':'punctuation'},
#                 keys={'start':'s', 'end':'e', 'label':'l'}):
#     try:
#         for ann in anns:
#             if ann['label'] in labels:
#                 ann['label'] = labels[ann['label']]
#             for key in ann.keys():
#                 if key in keys:
#                     ann[keys[key]] = ann.pop(key)
#     except KeyError:
#         pass
#     return anns

############################################################################
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str,
    help='file path of a CSV where 1st column is a unique ID, and 2nd is the text')

#group = parser.add_mutually_exclusive_group(required=True)
parser.add_argument('-d', '--destination', help='output directory',
                    default='data/result', type=str)

parser.add_argument('-t', '--tag', help='model tag',
                    default = None, type=str)

parser.add_argument('-m', '--model', help='pytorch model',
                    type=str)

args = parser.parse_args()

############################################################################

# fn_model = 'models/BiLSTM-round3-token_permutation-2020-01-31.pytorch'
# tag_model = 'round3-token_permutation-direct'
# 
# fn_model = 'models/BiLSTM-round1-token_permutation-60-epochs.pytorch'
# tag_model = 'round1-token_permutation-direct'
# 
# fn_model = 'models/BiLSTM-round2random-token_permutation-2020-02-28.pytorch'
# tag_model = 'round2r-token_permutation-direct'

os.makedirs(args.destination, exist_ok=True)

set_ = os.path.basename(args.input).split('.')[0]

if args.tag is None:
    tag_model = os.path.basename(args.model).split('.')[0]
else:
    tag_model = args.tag

print('MODEL TAG:', tag_model)

model = torch.load(args.model)

# get a tokenizer
tokenizer = spu.get_spacy_tokenizer(special_cases_file='special-cases-spacy.yaml')
reader = sgm.PosDatasetReader(index_sep='|', word_tag_delimiter='#%#')
predictor = sgm.AdvancedSentenceTaggerPredictor(model, 
                                            dataset_reader=reader,
                                            tokenizer=tokenizer)


# ## Label (predict on) the unlabelled data
df_input = pd.read_csv(args.input, index_col=0,  usecols=[0, 1], squeeze=True)

fn_bare_base =f'{args.destination}/{set_}'
fn_base =f'{fn_bare_base}-{tag_model}'


fn_gt = f'{fn_bare_base}-groundtruth.csv'
fn_prob = f'{fn_base}-prediction.csv'
fn_prediction = f'{fn_base}-prediction.json'
fn_parsed = f'{fn_base}-parsed.json'
fn_html = f'{fn_base}.html'
fn_meta = f'{fn_base}-meta.json'
with open(fn_meta, 'w') as fh:
    json.dump({kk:getattr(args, kk) for kk in dir(args) if not kk.startswith('_')}, fh)


print('number of records:', df_input.shape[0])
print('-'*40)
print('OUTPUT:')
print('META:', fn_meta)
print('PROBABILITIES TABLE:', fn_prob)
print('PREDICTIONS:', fn_prediction)
print('HTML:',fn_html)
print('PARSED:',fn_parsed)
print('-'*40)

df_input = df_input.loc[~df_input.index.isnull()]
df_input = df_input[~df_input.index.duplicated(keep='first')]
print('total entries:', df_input.shape[0])
print('empty entries:', df_input.map(lambda x: '  ' in x).sum())

############################################################################
# convert to  dask dataframe
dd_input = dd.from_pandas(df_input, npartitions=2*multiprocessing.cpu_count())

# run inference on the dask dataframe
df_logits = dd_input.apply(lambda x: predictor.predict(x.lower())['tag_logits'],
                          meta=('text', str)).compute()
df_logits = pd.concat({str(kk):vv for kk,vv in df_logits.map(pd.DataFrame).items()})
df_logits.index.names = ['title', 'token_num']
df_logits.columns = model.vocab._index_to_token['labels'].values()
df_logits.shape

da_logits = da.from_array(df_logits.values)
dd_prob_nonorm = np.exp(da_logits)

df_prob = dd_prob_nonorm/dd_prob_nonorm.sum(1, keepdims=True).compute()


df_prob = pd.DataFrame(np.asarray(df_prob),
                          index=df_logits.index, columns=df_logits.columns)

# SAVE PROBABILITIES
df_prob.to_csv(fn_prob)

# document_id, token_id, token, class1_score, class2_score, class3_score
#     note123,        1,    My,            0.02,        0.6,        0.38
#

# TOKENIZE SENTENCES
df_tokens_ = dd_input.apply(lambda x: predictor.tokenize_sentence(x),
                          meta=('text', str)).compute()
df_tokens = pd.concat({str(kk):vv for kk,vv in
                          df_tokens_.map(lambda row: pd.Series(row,)).items()})
df_tokens.name='token'
df_tokens = df_tokens.to_frame()

df_tokens['label'] = df_logits.idxmax(1)
df_tokens['entropy'] = (-np.log2(df_prob)*df_prob).sum(1)

df_tokens.index.names = [df_input.index.name, 'token_num']

df_tokens_wide = df_tokens.groupby(level=0).agg(list)
df_tokens_wide.shape

df_segments = spu.align_and_aggregate_tokens(df_input, df_tokens_wide, columns=['label', ])
if 'entropy' not in df_segments.columns:
    df_segments = pd.concat([df_segments, df_tokens['entropy'].groupby(level=0).max()], axis=1)
df_segments = df_segments.sort_values('entropy', ascending=False)


assert df_segments[df_segments.seq_annotations.isnull()].shape[0] == 0

df_segments['id'] = df_segments.index.tolist()

df_segments.seq_annotations = df_segments.seq_annotations.map(partial(sorted, key=lambda x: x['start']))

dict_segments = (df_segments
                 .sort_values('entropy', ascending=False)[['text', 'seq_annotations', 'id', 'entropy',]]
                 .to_dict( orient='records'))
print('number of records (json)', len(dict_segments))

# SAVE DOCUMENTS 
with open(fn_prediction, 'w') as fh:
    json.dump(dict_segments, fh)

#############################################
# CONVERT TO HTML
ds_html = df_segments.apply(spu.vis_segments_html, 1)

with open(fn_html, 'w') as fh:
     header = '<link rel="stylesheet" type="text/css" href="custom.css"/>\n<body>\n\n'
     footer = '</body>'
     fh.write(header)
     for _, line in ds_html.items():
         fh.write('<p>' + line + '</p><br/>\n')
     fh.write(footer)


#############################################
# Parsing

#df_segments.seq_annotations = df_segments.seq_annotations.map(partial(sorted, key=lambda x: x['start']))

single_letter_code = {'see':'e',
                      'letter':'l',
                      'notified':'n',
                      'source':'s',
                      'finding':'f',
                      'findings':'f',
                      'reference':'r',
                      'separator':'.',
                    'punctuation':'.'}

df_segments['segment_pattern'] = df_segments.seq_annotations.map(lambda x:''.join([single_letter_code[y['label']] for y in x]))


# segment_patterns
def all_or_none(x):
    return all(x) or not any(x)

def all_or_none_have_letter(captures):
    return all_or_none([y.strip('p')[0]=='l' for y in captures])


segment_patterns = df_segments['segment_pattern'].value_counts()
segment_patterns.name='counts'
# segment_patterns.index = segment_patterns.index.map(lambda x: ''.join(x))
segment_patterns = segment_patterns.to_frame()
segment_patterns['seq'] = segment_patterns.index.tolist()

pattern = regex.compile('^(?P<spec>(?P<letter>l)?\.?s\.?(f\.?e?\.?)+)+')
segment_patterns['match'] = segment_patterns['seq'].map(lambda x: pattern.match(x))
segment_patterns['spans'] = segment_patterns['match'].map(lambda mtch: mtch.spans('spec') if mtch is not None else [])
segment_patterns['cover'] = segment_patterns['spans'].map(lambda x: max((y[1] for y in x)) if len(x)>0 else 0)
# how many segments ('letters') are missed by the regex?
segment_patterns['missed'] = segment_patterns['seq'].map(len) - segment_patterns['cover']
segment_patterns['segmented'] = segment_patterns['match'].map(lambda mtch: '|'.join(mtch.captures('spec')) if mtch is not None else '')

#check whether all 
segment_patterns['consistent_letter'] = (segment_patterns['match']
                                         .map(lambda mtch: all_or_none_have_letter(mtch.captures('spec'))\
                                              if mtch is not None else False))
segment_patterns['well_segmented'] = (segment_patterns['consistent_letter'] & (segment_patterns['missed']<=1))
print('WELL SEGMENTED FRACTION', segment_patterns['well_segmented'].mean())
segment_patterns.shape[0]


if 'segmented' in df_segments:
    df_segments.drop(['segmented', 'match'], axis=1, inplace=True)
df_segments = pd.merge(df_segments, segment_patterns[segment_patterns.well_segmented][['segmented','match']],
                          how='left',
                          left_on='segment_pattern', right_index = True,)



(df_segments
    .sort_values('entropy', ascending=False)[['text', 'seq_annotations', 'id', 'entropy',]]
    .to_json(fn_parsed, orient='records'))


