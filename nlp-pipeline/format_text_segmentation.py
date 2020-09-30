#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import csv
import json
import yaml
import segm_prep_utils as spu

#from pathlib import Path
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('input', type=str,
    help='file path of the JSONL-formatted labelled documents exported from Doccano')
parser.add_argument('-d', '--destination', help='output',
                    type=str)
                    
parser.add_argument('--tok-sep', help='token-label separator sequence in output',
                    default = '#%#', type=str)

parser.add_argument('-s', '--start', help='start field in JSON',
                    default = 'start_offset', type=str)

parser.add_argument('-e', '--end', help='end field in JSON',
                    default = 'end_offset', type=str)

parser.add_argument('-l', '--limit', help='cut off at this number of documents',
                    default=None, type=int)

parser.add_argument('-a', '--after', help='take all documents after this index',
                    default=None, type=int)

parser.add_argument('--dask', action="store_true", default=False,
                    dest='dask', help="process with dask library")

#parser.add_argument('--no-dask', action="store_false", default=True,
#                    dest='dask', help="process without dask library")

args = parser.parse_args()



table = pd.read_json(args.input, lines=True)
range_ = None
if args.limit or args.after:
    sl = slice(args.after, args.limit, None)
    range_ = ((args.after if args.after else 0),
              (args.limit if args.limit else len(table))
             )
    table = table[sl]

table.set_index('title', inplace=True)

destination = args.destination if \
    args.destination is not None else \
    args.input.replace('.jsonl', '').replace('.json', '') + \
        ('-indices-{}-{}'.format(*range_) if range_ is not None else '') + \
        '.csv'

print('SAVING TO:', destination)

with open('special-cases-spacy.yaml') as fh:
       special_cases = yaml.load(fh)


## conversion / formatting per se
spacy_tokenizer = spu.get_spacy_tokenizer(special_cases_file='special-cases-spacy.yaml')

# slow process
convert = lambda x: spu.annotation_json2septext(
    x['text'], x['seq_annotations'],
    sep=args.tok_sep,
    tokenize=spacy_tokenizer, 
#   sep_label='punctuation',
    start=args.start,
    end=args.end,
    ).lower()

if args.dask:
    import dask
    import dask.dataframe as dd
    import dask.array as da
    import multiprocessing
    table_da = dd.from_pandas(table, npartitions=multiprocessing.cpu_count())
    dfplain = table_da.apply(convert, axis=1).compute()
else:
    dfplain = table.apply(convert, axis=1)


dfplain.to_frame().to_csv(destination, quoting=csv.QUOTE_NONE,
                         sep='|',quotechar='',
                         index=True, header=False, )

