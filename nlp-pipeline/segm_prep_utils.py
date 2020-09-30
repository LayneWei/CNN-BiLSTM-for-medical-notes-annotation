import sys
import re
import pandas as pd
import numpy as np
from collections import OrderedDict
from typing import List, Set
from warnings import warn
#from numba import jit

def annotation_json2html(txt, anns,
                   start = 'start',
                   end = 'end',
                   label='label'):
    anns = sorted(anns, key = lambda x: x[start])
    prev_end = 0
    html_ = ''
    for ann in anns:
        label_ = ann[label]
        if (ann[start]-prev_end)>0:
            html_+= ('<span>' + txt[prev_end:ann[start]] + '</span>')
        html_+= (f'<span class={label_}>' + txt[ann[start]:ann[end]] + '</span>')
        prev_end = ann[end]
    if prev_end < len(txt):
        html_+= ('<span>' + txt[prev_end:len(txt)] + '</span>')
    return html_

#@jit
def annotation_json2septext(txt:str, anns:List[dict],
                  start = 'start',
                  end = 'end',
                  label='label',
                  sep='#%#',
                  #punctuation=''',.:;()#"''',
                  tokenize = lambda x: normalize_punctuation(x, ''',.:;()#"''').split(' '),
                  sep_label = 'separator',
                  LETTERS= re.compile(r'[a-zA-Z]+'),
                  ) -> str:
    """convert json to token-by-token labelled format,
    e.g.: 'Barak#%#person Obama#%#person visited#%#other EU#%#location'

    warning: current implementation is slow
    """
    anns = sorted(anns, key = lambda x: x[start])
    prev_end = 0
    prev_piece = ''
    html_ = ''
    token_label_pairs = []
    for ann in anns:
        if (ann[start]-prev_end)>0:
            piece = txt[prev_end:ann[start]].strip()
            tokens = tokenize(piece)
            token_label_pairs.extend([(tok, sep_label) for tok in tokens])
            prev_piece = piece
            prev_label = sep_label

        label_ = ann[label]
        piece = txt[ann[start]:ann[end]]

        prev_match_inv = LETTERS.match(prev_piece[::-1])
        this_match = LETTERS.match(piece)

        # handle within-token splits
        if len(prev_piece) and \
            prev_match_inv and \
            this_match:
            len_after = this_match.span()[1]
            len_before = prev_match_inv.span()[1]
            warn(f'potentially wrong split: "{prev_piece}|{piece}"')

            if len_after >= len_before:
                # remove previous pair and append the dangling part from previous piece to the new one
                token_label_pairs.pop()
                piece = prev_piece[-len_before:] + piece
                warn(f'{len_before} < {len_after}\t fixing to previous:'
                     f'\t"{prev_piece[-len_before:]}|{piece[:len_after]}"\n'
                     f'\t"{piece}"')
            else:
                # this handles only most straightforward all-alphabetic tokens
                token_label_pairs.pop()
                prev_piece_dangle = prev_piece[-len_before:] + piece[:len_after]
                token_label_pairs.append( (prev_piece_dangle, prev_label) )
                warn(f'{len_before} > {len_after}\t fixing to next:'
                     f'\t"{prev_piece[-len_before:]}|{piece[:len_after]}"\n'
                     f'\t"{prev_piece_dangle}"')


        tokens = tokenize(piece)
        token_label_pairs.extend([(tok, label_) for tok in tokens])
        ## piece = normalize_punctuation(piece, punctuation)
        prev_end = ann[end]
        prev_piece = piece
        prev_label = label_

    if prev_end != len(txt):
        piece = txt[prev_end:len(txt)]
        tokens = tokenize(piece)
        token_label_pairs.extend([(tok, sep_label) for tok in tokens])

    if sep is None:
        return token_label_pairs

    html_ = ' '.join([ '{}{}{}'.format(tok, sep, lbl) for tok, lbl in token_label_pairs])
    html_ = clean_whitespace(html_, sep=sep)
    return html_.strip()


def clean_whitespace(anntxt, sep = '#%#'):
    return ' '.join([x for x in anntxt.split(' ') if not x.startswith(sep)])


def clean_rename_dict(xdict, 
                     rename={'start_offset':'start', 
                             'end_offset':'end'},
                     remove=['user_id', 'prob', 'manual']):
    "inplace renaming of dictionary keys"
    try:
        for kk,vv in rename.items():
            xdict[vv] = xdict.pop(kk)
        for kk in remove:
            xdict.pop(kk)
        return xdict
    except KeyError as ee:
        print(xdict)
        raise ee


def label_words(tokens, label_, sep='#%#', tokenize=lambda x: x.split(' ')):
    "Deprecated"
    return ' '.join([ '{}{}{}'.format(x, sep, label_) for x in tokens])


def normalize_punctuation(qry, separators = r''',.:;()#'"'''):
    # ,.:;()#\"\'\-
    #separators = re.escape(separators)
    for _ in range(3):
        qry = re.sub(re.compile(r'([^\ ])([{}])([^\d]|$)'.format(separators)), r'\1 \2\3', qry)
        qry = re.sub(re.compile(r'([^\d])([{}])([^\ ]|$)'.format(separators)), r'\1\2 \3', qry)
    return qry


def strip_punctuation(piece, punctuation=''' ,.:;()#'''):
    match_ = re.match(re.escape(piece.strip(punctuation)), piece)
    start_m, end_m = match_.span()
    piece[start_m:end_m]
    return piece[:start_m], piece[start_m:end_m], piece[end_m:]




def get_color_html(txt, c):
#     source = "<h1 style='color: rgb({0})'>{1}</h1>".format(c, txt)
    source = "<span style='color:{0}'>{1}</span>".format(c, txt)
    return source


def get_classy_html(txt, label):
    return f'<span class={label}>{txt}</span>'


def annotated_html(txt, anns,
                   start = 'start',
                   end = 'end',
                   label='label'):
    anns = sorted(anns, key = lambda x: x[start])
    prev_end = 0
    html_ = ''
    for ann in anns:
        prev_end
        label_ = ann[label]
    #     print('diff', ann['start_offset']-prev_end)
        if (ann[start]-prev_end)>0:
            html_+= ('<span>' + txt[prev_end:ann[start]] + '</span>')
        html_+= (f'<span class={label_}>' + txt[ann[start]:ann[end]] + '</span>')
        prev_end = ann[end]
    return html_


def vis_segments_html(row,
                      start='start', end='end', 
                      label='label', title='id'):

    return ('<div class="title">' + row[title] 
            + '</div>&emsp;&emsp;<div class="text">'
            + annotation_json2html(row['text'], row['seq_annotations'],
                                   start = start,
                                   end = end,
                                   label=label)
            + '</div>'
            )


def agg_segments_json(original_texts: pd.Series, 
                      segment_positions:pd.DataFrame,
                      name = 'seq_annotations',
                      ):
    """Create a table with segments and their annotations 
    represented as one row per document,
    which can be easily exported to JSON / JSONL
    """
    # reshape original table
    title = segment_positions.index.names[0]
    ds_json_segments = pd.Series({kk:vv.to_dict(orient='records') 
                                  for kk,vv in segment_positions.groupby(title)})
    ds_json_segments.name = name
    
    original_texts.name = 'text'
    ds_json_segments = pd.concat([ds_json_segments, original_texts], 
                                 axis=1, join='inner')
    
    ds_json_segments['id'] = ds_json_segments.index
    ds_json_segments.index.name = title
    return ds_json_segments



def align_token_sentence(txt_:str, 
                         tokens:List[str],
                         **kwargs: List,
                         ):
    verbose=False
    txt_ = txt_.lower()
    start = 0
    positions = []
    for nn, tok in enumerate(tokens):
        tok_compiled = re.compile(re.escape(tok.lower()))
        try:
            match = tok_compiled.search(txt_[start:])
        except Exception as ee:
            print(f'token: "{tok}"\ttext: {txt_[start:]}')
            raise ee
        if not match:
            raise ValueError(f'no matches for "{tok}" in: {txt_[start:]}')
        start_m, end_m = match.span()
        row = {'start':start+start_m, 'end':start+end_m, 'token': tok}
        if verbose:
            print(tok, start+start_m, start+end_m, txt_[start+end_m:],'\n', sep='|')
        if len(kwargs):
            for kk, vv in kwargs.items():
                if isinstance(vv, str):
                    raise ValueError('provided a string instead of list in keyword args:\n'
                                    'key:{}\targument:{}'.format(kk, vv))
            #row = (start+start_m, start+end_m, tok, labels[nn])
                row[kk] = vv[nn]

        positions.append(row)
        start = start+end_m
    return positions

#def aggregate_tokens(df_token_start_end_label: pd.DataFrame,
#                        columns = ['label'], index=None):

def align_and_aggregate_tokens(original_texts: pd.Series,
                        tokens_labels: pd.DataFrame,
                        columns = ['label'], index=None):

    df_token_start_end_label = align_with_original_text(original_texts, tokens_labels,
                                                        columns=columns)
    if index is None:
        index = df_token_start_end_label.index.names[0]
    df_segments = pd.concat([original_texts,
                            token_table_to_dictionaries(df_token_start_end_label, index=index)],
                           axis=1, sort=False)
    df_segments.index.name = original_texts.index.name
    return df_segments


def token_table_to_dictionaries(tokens_long, label_col='label', index='title'):
    result = tokens_long.groupby(index).apply(
        lambda x: aggregate_segments(x[['start', 'end', label_col]],
                                     label_col=label_col).to_dict(orient='records'))
    result.name = 'seq_annotations'
    return result


def align_with_original_text(original_texts: pd.Series,
                             token_labels:pd.DataFrame,
                             columns = ['label']) -> pd.DataFrame:
    """Align tagged tokens with the original text.

    Input:
    - original_texts:  a series with original text in each row
    - token_labels:    a _nested_ _wide_ dataframe with lists of tokens and labels 
                       per each row (report)

    Returns a long table with one row per each token,
      indexed as (title, token_id),
      where title is a document title
    """
    original_texts.name = 'text'

    dfm = pd.merge(token_labels[['token']+columns], original_texts,
            left_index=True, right_index=True).dropna()

    index_name = token_labels.index.name
    index_name = index_name or original_texts.index.name
    dfm.index.name = index_name

    try:
        columns.remove('token')
    except:
        pass

    token_positions_wide = dfm \
        .apply(lambda row: align_token_sentence(row['text'], row['token'], 
                                                **{col:row[col] for col in columns}),
               axis=1)

    token_positions_wide.index.name = original_texts.index.name

    token_positions = pd.concat(token_positions_wide
                                .map(lambda row: pd.DataFrame(row))
                                .to_dict())
    token_positions.index.names = [token_positions_wide.index.name, 'token_id']
    return token_positions



def aggregate_segments(doc: pd.DataFrame, label_col='prediction', 
                       label_dict={}):
    """Aggregates segments of tokens with the same label within each document.
    doc: pd.DataFrame with columns: ['label', 'start', 'end']
    """
    #labels = [row[label_col] for row  in doc]
    labels = doc[label_col]
    if len(label_dict)==0:
        label_dict = {vv:kk for kk,vv in enumerate(list(set(labels)))}

    breakpoints = (labels.map(lambda x:label_dict[x]).diff()!=0).to_numpy() \
                  .nonzero()[0].tolist()
    breakpoints = np.asarray(breakpoints + [doc.shape[0]])
    df_segments = pd.DataFrame(OrderedDict([
               ('start', doc.iloc[breakpoints[:-1]]['start'].values),
               ('end', doc.iloc[breakpoints[1:]-1]['end'].values),
               ('label', doc.iloc[breakpoints[:-1]][label_col].values)]),
              )
    return df_segments

def get_spacy_tokenizer(special_cases_file='special-cases-spacy.yaml', verbose=False):
    import yaml
    from allennlp.data.tokenizers import SpacyTokenizer
    #from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter as SpacyTokenizer
    #nlp = English('en_core_web_sm')
    

    with open('special-cases-spacy.yaml') as fh:
        special_cases = yaml.load(fh)

    tokenizer_obj = SpacyTokenizer(language='en_core_web_sm', pos_tags=True)
    for case in special_cases:
        if verbose:
            print(*case)
        tokenizer_obj.spacy.tokenizer.add_special_case(*case)

    #return tokenizer_obj.split_words
    return tokenizer_obj.tokenize


# deprecated; see annotation_json2septext
# earlier: annotated_text
# def annotate_tokens(txt, anns,
#                   start = 'start_offset',
#                   end = 'end_offset',
#                   label='label',
#                   sep='#%#',
#                   punctuation = ''',.:;()#''"'''):
#     anns = sorted(anns, key = lambda x: x[start])
#     prev_end = 0
#     html_ = ''
#     for ann in anns:
#         prev_end
#         label_ = ann[label]
#         if (ann[start]-prev_end)>0:
#             piece = txt[prev_end:ann[start]].strip()
#             html_+= ' '  + label_words(piece, 'separator', sep).strip()
#         piece = txt[ann[start]:ann[end]]
#         piece = normalize_punctuation(piece, separators=punctuation)
#         html_+= ' '+ label_words(piece, label_, sep)
#         prev_end = ann[end]
#     return html_.strip()

    seq_anns = sorted(seq_anns, key=lambda x: x['start'])
    prev_ann = dict(start=-1, end=-1, label='')
    for ann in seq_anns:
    #     print(prev_ann['end'],ann['start'])
        if prev_ann['end']>ann['start']:
#             print(prev_ann, ann, sep='\t')
            return True
#             print(prev_ann, ann, sep='\t')
#             if prev_ann['label'] == 'punctuation':
#                 return True
#             if ann['label'] == 'punctuation':
#                 return True
#                 prev_ann['end'] = ann['start']
        prev_ann = ann
    return False


# not really much used:
def check_ann_overlap_no_punct(seq_anns):
    seq_anns = sorted(seq_anns, key=lambda x: x['start'])
    prev_ann = dict(start=-1, end=-1, label='')
    for ann in seq_anns:
    #     print(prev_ann['end'],ann['start'])
        if prev_ann['end']>ann['start']:
#             print(prev_ann, ann, sep='\t')
            return True
#             print(prev_ann, ann, sep='\t')
            if (prev_ann['label'] == 'punctuation') or ann['label'] == 'punctuation':
                return False
            else:
                return True
#                 prev_ann['end'] = ann['start']
        prev_ann = ann
    return False

