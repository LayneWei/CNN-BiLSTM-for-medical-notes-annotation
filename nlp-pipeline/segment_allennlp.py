#!/usr/bin/env python
# coding: utf-8
from typing import Iterator, List, Dict
import torch
import torch.optim as optim

import sys
import numpy as np
import yaml
import re
from functools import reduce
from functools import partial
from collections import deque, OrderedDict

from allennlp.data.dataset_readers import DatasetReader, SequenceTaggingDatasetReader
from allennlp.data.samplers import BucketBatchSampler as BucketIterator
from allennlp.data import DataLoader as DataIterator
from allennlp.data import Batch

#from allennlp.data.iterators.data_iterator import DataIterator
#from allennlp.data.dataset import Batch
#from allennlp.data.iterators import BucketIterator


from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common.util import lazy_groups_of, add_noise_to_dict_values
from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
# deprecated:
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder

from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer
#from allennlp.training.learning_rate_schedulers import LearningRateWithoutMetricsWrapper
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.predictors.predictor import Predictor

torch.manual_seed(1)

import pandas as pd
import segm_prep_utils as spu

from copy import deepcopy
from overrides import overrides

from typing import List, Tuple, Iterable, cast, Dict, Deque, Callable

import logging
import random



logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PUNCTUATION = set(list('.,:;()'))

class LstmTagger(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 loss_params={}) -> None:
        """
        loss_params : see help(allennlp.nn.util.sequence_cross_entropy_with_logits)

        """
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = partial(sequence_cross_entropy_with_logits, **loss_params)
        
    def forward(self,
                sentence: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(sentence)
        embeddings = self.word_embeddings(sentence)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if labels is not None:
            #fix
            if not hasattr(self, 'loss_function'):
                self.loss_function = sequence_cross_entropy_with_logits
            # self.accuracy(tag_logits, labels, mask)
            output["loss"] = self.loss_function(tag_logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}



class PosDatasetReader(SequenceTaggingDatasetReader):
# class PosDatasetReader(DatasetReader):
    """
    DatasetReader for PoS tagging data, one sentence per line, like

        The###DET dog###NN ate###V the###DET apple###NN
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 word_tag_delimiter:str = '###',
                 index_sep: str = None,
                 remove_tokens:List[str] = [],
                 rename_labels:Dict = {}) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.index_sep = index_sep
        self.sep = word_tag_delimiter
        self.remove_tokens = remove_tokens
        self.rename_labels = rename_labels
        

    def text_to_instance(  # type: ignore
        self, tokens: List[Token], tags: List[str] = None, idx:str = None
    ) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """

        fields: Dict[str, Field] = {}
        sequence = TextField(tokens, self._token_indexers)
        fields["tokens"] = sequence
        fields["metadata"] = MetadataField({"words": [x.text for x in tokens]})
        if tags is not None:
            fields["tags"] = SequenceLabelField(tags, sequence)
        inst = Instance(fields)
        inst.index = idx
        return inst

    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for nn, line in enumerate(f):
                try:
                    line = line.strip()
                    if self.index_sep is not None:
                        idx, line = line.split(self.index_sep)
                    else:
                        idx = None
                    pairs = line.strip().split()
                    sentence, tags = zip(*(pair.split(self.sep) for pair in pairs))
                    if len(self.remove_tokens):
                        sentence, tags = list(zip(*[(word, label) for word, label in
                            zip(sentence, tags)
                            if word not in self.remove_tokens]))
                    if self.rename_labels:
                        tags = [(tt if (tt not in self.rename_labels) else self.rename_labels[tt])
                                for tt in tags]
                    tokens = [Token(word) for word in sentence]
                    yield self.text_to_instance(tokens, tags, idx)

                except ValueError as ee:
                    print('error in :\n', line)
                    raise ee

    @classmethod
    def long_table(cls, dataset):
        return pd.DataFrame(sum((list(zip(
                                      [inst.index]*len(inst.fields['labels']), 
                                      range(len(inst.fields['labels'])), 
                                      list(inst.fields['labels']))
                                     ) for inst in dataset),
                                []),
                            columns=['title', 'token_index','label']
                            ).set_index(['title', 'token_index'])['label']
                

# class LstmTagger(Model):
#     def __init__(self,
#                  word_embeddings: TextFieldEmbedder,
#                  encoder: Seq2SeqEncoder,
#                  vocab: Vocabulary) -> None:
#         super().__init__(vocab)
#         self.word_embeddings = word_embeddings
#         self.encoder = encoder
#         self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
#                                           out_features=vocab.get_vocab_size('labels'))
#         self.accuracy = CategoricalAccuracy()
#         
#     def forward(self,
#                 sentence: Dict[str, torch.Tensor],
#                 labels: torch.Tensor = None) -> torch.Tensor:
#         mask = get_text_field_mask(sentence)
#         embeddings = self.word_embeddings(sentence)
#         encoder_out = self.encoder(embeddings, mask)
#         tag_logits = self.hidden2tag(encoder_out)
#         output = {"tag_logits": tag_logits}
#         if labels is not None:
#             self.accuracy(tag_logits, labels, mask)
#             output["loss"] = sequence_cross_entropy_with_logits(tag_logits, labels, mask)
# 
#         return output
#     def get_metrics(self, reset: bool = False) -> Dict[str, float]:
#         return {"accuracy": self.accuracy.get_metric(reset)}


def remove_punctuation(reader, inst, punctuation=PUNCTUATION):
    words,  labels = list(zip(*[(word, label) for word, label in
        zip(inst.fields['sentence'], inst.fields['labels'].labels, )
        if word.text not in punctuation]))
    sentence = TextField(words, reader.token_indexers)
    label_field = SequenceLabelField(labels=labels, sequence_field=sentence)
    inst_out =  Instance({"sentence": sentence, 'labels':label_field})
    if hasattr(inst, 'index'):
        inst_out.index = inst.index
    return inst_out


def text_to_instance(sentence: List[str], tags: List[str] = None,
                     idx:str=None, token_indexers=None) -> Instance:
    token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    tokens = [Token(word) for word in sentence]
    sentence_field = TextField(tokens, token_indexers)
    fields = {"sentence": sentence_field}

    if tags:
        label_field = SequenceLabelField(labels=tags, sequence_field=sentence_field)
        fields["labels"] = label_field
    inst = Instance(fields)
    inst.index = idx
    return inst


def mutate_token(inst, frequency=0.5):
    words,  labels = list(zip(*[(word.text, label) for word, label in
                         zip(inst.fields['sentence'], inst.fields['labels'].labels, )
                         if word.text not in PUNCTUATION]))

    words = list(words)
    if np.random.rand()>frequency:
        words[np.random.randint(len(words))] = 'unk'
    return text_to_instance(words,  labels)


def permute_token(inst, frequency=0.5,
                  exclude={'separator', 'letter'}):
    if np.random.rand()<frequency:
        return inst
    
    words,  labels = list(zip(*[(word.text, label) for word, label in
                          zip(inst.fields['sentence'], inst.fields['labels'].labels, ) 
                               ]))

    words = list(words)
    valid_positions = np.where([((lbl not in exclude)
                                 and (wrd not in PUNCTUATION | {'see', 'slide'})) \
                                for lbl, wrd in zip(labels, words)])[0]
    
    pos_from = np.random.choice(valid_positions)
    valid_positions = np.where([((lbl not in exclude |{labels[pos_from]})
                                 and (wrd not in PUNCTUATION | {'see', 'slide'})) \
                                for lbl, wrd in zip(labels, words)])[0]
    if len(valid_positions)>0:
        pos_to = np.random.choice(valid_positions)
    #     print(words[pos_from], words[pos_to])
    #     print(labels[pos_from], labels[pos_to])
        tmp = words[pos_from]
        words[pos_from] = words[pos_to]
        words[pos_to] = tmp
    return text_to_instance(words,  labels)



def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]


#@DataIterator.register("advanced_bucket")
class AdvancedBucketIterator(DataIterator):
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).

    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
        "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
        "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
        "num_token_characters" of the "sentence1" field.  TODO(mattg): we should have some
        documentation somewhere that gives the standard padding keys used by different fields.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.

        Note that if you specify ``max_instances_in_memory``, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        See :class:`BasicIterator`.
    """

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 preprocess: Callable[[Instance], Instance] = None) -> None:
        if not sorting_keys:
            raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self.preprocess = preprocess
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first

    #@overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        if self.preprocess:
            instances = list(map(self.preprocess, instances))
        else:
#             print()
            instances = list(instances)
        for instance_list in self._memory_sized_lists(instances):

            instance_list = sort_by_padding(instance_list,
                                            self._sorting_keys,
                                            self.vocab,
                                            self._padding_noise)

            batches = []
            excess: Deque[Instance] = deque()
            for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                for possibly_smaller_batches \
                  in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                    batches.append(Batch(possibly_smaller_batches))
            if excess:
                batches.append(Batch(excess))

            move_to_front = self._biggest_batch_first and len(batches) > 1
            if move_to_front:
                # We'll actually pop the last _two_ batches, because the last one might not be full.
                last_batch = batches.pop()
                penultimate_batch = batches.pop()
            if shuffle:
                # NOTE: if shuffle is false, the data will still be in a different order
                # because of the bucket sorting.
                random.shuffle(batches)
            if move_to_front:
                batches.insert(0, penultimate_batch)
                batches.insert(0, last_batch)

            yield from batches


#@Predictor.register('advanced-sentence-tagger3')
class AdvancedSentenceTaggerPredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single set of tags for it.  In particular, it can be used with
    the :class:`~allennlp.models.crf_tagger.CrfTagger` model
    and also
    the :class:`~allennlp.models.simple_tagger.SimpleTagger` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader,
                tokenizer=Callable[[str], List]) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = tokenizer
        # SpacyTokenizer(language='en_core_web_sm', pos_tags=True).split_words

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence" : sentence})


    @property
    def labels(self):
        return list(self._model.vocab._index_to_token['labels'].values())

    @property
    def label_dict(self):
        return {kk:vv for vv,kk in enumerate(self.labels)}


    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = self._tokenizer(sentence)
        instance = self._dataset_reader.text_to_instance(tokens)
        return instance


    def tokenize_sentence(self, x):
        return list(map(str, self._json_to_instance({'sentence':x}).fields['sentence']))

    def tokenize_sentence_series(self, df_all_nolab):
        """one-liner failed;(
         pd.concat({str(kk):vv for kk,vv in \
                                   df_all_nolab.map(lambda x: tokenize_sentence(x)).map(pd.Series).items()}
         ).to_frame()
        """
        tmp = {}
        for kk,vv in df_all_nolab.items():
            try:
                tmp[kk] = pd.Series(self.tokenize_sentence(vv))
            except Exception as ee:
                print(kk)
                raise ee
        dfres_tokens = pd.concat(tmp).to_frame()
        dfres_tokens.columns = ['token']
        dfres_tokens["token"] = dfres_tokens["token"].map(str)
        dfres_tokens.index.names = ['title', 'token_num']
        return dfres_tokens

    def get_logits(self, df_sentences, preprocess = lambda x: x.lower()):
        preprocess = preprocess if preprocess is not None else (lambda x:x)
        dfres_logits = df_sentences.apply(lambda x: self.predict(x)['tag_logits'])
        dfres_logits = pd.concat({str(kk):vv for kk,vv in dfres_logits.map(pd.DataFrame).items()})
        dfres_logits.index.names = ['title', 'token_num']
        dfres_logits.columns=self._model.vocab._index_to_token['labels'].values()
        return dfres_logits

    def predict_logits_table(self, text_list):
        """given a list or pandas.Series of documents,
        predict logits and return a long table of logits per token
        """
        if isinstance(text_list[0], Instance): # dataset
            dfres = pd.DataFrame(list(map(self.predict_on_sentence, text_list)))
            dfres.index = [x.index for x in text_list]
            dfres_logit_lists = dfres['logit']
        elif isinstance(text_list[0], str):
            text_list = pd.Series(text_list)
            dfres_logit_lists = text_list.apply(lambda x: self.predict(x)['tag_logits'])
            #dfres_logits = pd.concat({str(kk):vv for kk,vv in dfres_logits.map(pd.DataFrame).items()})

        dfres_logits = self.logit_list_to_table(dfres_logit_lists)
        return dfres_logits

    def logit_list_to_table(self, logits_series):
        """
        input: a pandas.Series of lists of logits for each sentence
        """
        dfres_logits = pd.concat({str(kk):vv for kk,vv in 
                                    logits_series.map(pd.DataFrame).items()})
        dfres_logits.index.names = ['title', 'token_num']
        dfres_logits.columns = self._model.vocab._index_to_token['labels'].values()
        return dfres_logits


    def predict_on_sentence(self, inst):
        """predict on a document represented as an allennlp.data.Instance object
        """
        model = self._model

        # import ipdb; ipdb.set_trace() # BREAKPOINT
        logits_ = model.forward_on_instance(inst)['tag_logits']
        labels_ = [model.vocab.get_token_from_index(i, 'labels') for i in logits_.argmax(1)]
        # onehot_ = list(map(lambda x :[(x in kk) for kk in model.vocab._index_to_token['labels'].values()],
        #            labels_))
        # assert len(onehot_) == len(logits_)

        txt = inst['sentence'].tokens
        txt = [x.text for x in txt]
        return {
             'token': txt,
   #          'onehot': onehot_,
             'logit': logits_,
             'label': labels_,
        }

def _sort_by_start(seq, key='start'):
    return sorted(seq, key=lambda x: x[key])


def merge_annotation2json(tokens, sentences):
    token_positions = align_token_sentences_df(tokens['token'], sentences)
    dfres_comb = pd.concat([token_positions, tokens[['label']]], axis=1, join='inner')

    label_dict = {kk:vv for vv,kk in enumerate(tokens['label'].unique())}

    df_segments = dfres_comb.groupby(level=0).apply(
                    partial(get_predicted_segments, label_dict=label_dict, label_col='label'))
    ds_json_segments = pd.Series({kk:_sort_by_start(vv.to_dict(orient='records')) for kk,vv in df_segments.groupby('title')})
    # ds_json_segments.name = 'seq_annotations'
    ds_json_segments = pd.concat({'seq_annotations':ds_json_segments,
                                  'text':sentences}, 
                                  axis=1, join='inner')
    ds_json_segments['id'] = ds_json_segments.index
    return ds_json_segments

# legacy; deprecated
def align_token_sentences_df(tokens, sentences):
    """DEPRECATED
    Aligns tokens to original sentences
    # Input:
        tokens   :    a Series with MultiIndex('title', 'token_index')
        sentence :    a Series with a simple Index('title')
    # Output
        token_positions: a Series indicating position of each token  
                      within the original sentence
                      with MultiIndex('title', 'token_index')
                      and columns ['start', 'end', 'token']
    """
    token_positions = []
    for tl_, snt_ in tokens.groupby(level='title'):
        try:
            txt_ = sentences.loc[tl_,].lower()
            positions_ = align_token_sentence(snt_.map(re.escape), txt_)
            positions_ = pd.DataFrame(positions_, index=snt_.index, columns=['start','end'])
            positions_ = positions_.merge(snt_.to_frame(), right_index=True, left_index=True,)
            token_positions.append(positions_)
        except Exception as ee:
            print('at:\t%s'%tl_, file=sys.stderr)
            print('sentence:\t%s'%txt_, file=sys.stderr)
            raise ee
    token_positions = pd.concat(token_positions)
    return token_positions


def align_token_sentence(tokens, txt_, verbose=False):
    start = 0
    positions = []
    for tok in tokens:
        match = re.search(tok, txt_[start:])
        if not match:
            raise ValueError('no matches:\t[%s]\tin\t[%s]' % (tok, txt_[start:]))
        start_m, end_m = match.span()
        if verbose:
            print(tok, start+start_m, start+end_m, txt_[start+end_m:],'\n')
        positions.append((start+start_m, start+end_m))
        start = start+end_m
    return positions


def get_predicted_segments(snt, label_dict={}, label_col='prediction'):
    if len(label_dict)==0:
        label_dict = {kk:vv for vv,kk in enumerate(snt[label_col].unique())}
    breakpoints = (snt[label_col].map(lambda x:label_dict[x]).diff()!=0).nonzero()[0].tolist()
    breakpoints = np.asarray(breakpoints + [snt.shape[0]])
    df_segments = pd.DataFrame(OrderedDict([('start', snt.iloc[breakpoints[:-1]]['start'].values),
               ('end', snt.iloc[breakpoints[1:]-1]['end'].values), 
               ('label', snt.iloc[breakpoints[:-1]][label_col].values)]),
              )
    return df_segments


def get_color_html(txt, c):
#     source = "<h1 style='color: rgb({0})'>{1}</h1>".format(c, txt)
    source = "<span style='color:{0}'>{1}</span>".format(c, txt)
    return source


def get_classy_html(txt, label):
    return f'<span class={label}>{txt}</span>'


def annotate_html(txt, anns,
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


def predict_on_sentence(inst):
    logits_ = model.forward_on_instance(inst)['tag_logits']
    labels_ = [model.vocab.get_token_from_index(i, 'labels') for i in logits_.argmax(1)]
#     sentence_indices.extend([inst.index]*len(logits_))
#     token_indices.extend(np.arange(len(logits_)))
    onehot_ = list(map(lambda x :[(x in kk) for kk in model.vocab._index_to_token['labels'].values()],
               labels_))
    assert len(onehot_) == len(logits_)
    
    txt = inst['sentence'].tokens
    txt = [x.text for x in txt]
#     set_tokens.extend([(inst.index, ii, x) for ii,x in enumerate(txt)])
#     html_ = [get_classy_html(kk, vv) for kk,vv in zip(txt, labels_)]
    return {
         'token': txt,
#          'onehot': onehot_,
         'logit': logits_,
         'label': labels_,
    }


def get_ground_truth(dataset):
    return pd.DataFrame(sum((list(zip(
                                [inst.index]*len(inst.fields['labels']), 
                                 range(len(inst.fields['labels'])), 
                                 list(inst.fields['labels']))) for inst in dataset), 
                                []), columns=['title', 'token_index','label']
                              ).set_index(['title', 'token_index'])['label']

#####################################################################

if __name__ == '__main__':
    from allennlp.data.tokenizers import SpacyTokenizer
    #from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter as SpacyTokenizer
    from torch.optim.lr_scheduler import MultiStepLR

    # read datasets

    reader = PosDatasetReader(index_sep='|', word_tag_delimiter='#%#')
    # reader = SequenceTaggingDatasetReader(word_tag_delimiter='#%#')
    fns = {}
    datasets = {}
    for set_ in ['train', 'val', 'dev_test', 'test', 'all']:
        fns[set_] = 'data/ephi_reports-1908-entries-2019-01-14-formatted-{}.csv'.format(set_)
        datasets[set_] = reader.read(fns[set_],)
        print(set_, list(reduce(lambda x,y: x|y, (set(inst.fields['labels'])  for inst in  datasets[set_] ))), sep='\t')


    dataset_train_no_punct = [(remove_punctuation(x, PUNCTUATION)) for x in datasets['train']]
    dataset_all_no_punct = [(remove_punctuation(x,  PUNCTUATION)) for x in datasets['all']]
    # dataset_train_permuted = [(permute_token(x, 1.0)) for x in datasets['train']]


    EMBEDDING_DIM = 16
    HIDDEN_DIM = 6
    DROPOUT = 0.5
    batch_size=2

    vocab = Vocabulary.from_instances(reduce(lambda x,y: x+y, datasets.values()) )
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM,
                                               batch_first=True,
                                               bidirectional=True,
                                               dropout=DROPOUT))

    model = LstmTagger(word_embeddings, lstm, vocab)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = AdvancedBucketIterator(batch_size=batch_size,
                                      sorting_keys=[("sentence", "num_tokens")],
                                      preprocess=partial(permute_token, frequency=0.2),
                                     )
    iterator.index_with(vocab)

    val_iterator = AdvancedBucketIterator(batch_size=2, 
                                      sorting_keys=[("sentence", "num_tokens")],
                                     )
    val_iterator.index_with(vocab)



    USE_CUDA = True
    if USE_CUDA:
        model = model.cuda()

    num_epochs = 30


    learning_rate_scheduler = LearningRateWithoutMetricsWrapper(MultiStepLR(optimizer,
                                                                            [10, 20, 40],
                                                                            gamma=0.25, last_epoch=-1))

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      validation_iterator = val_iterator,
                      #train_dataset=dataset_train_no_punct + datasets['train'],
                      train_dataset = dataset_all_no_punct+ datasets['all'],
    #                   validation_dataset=datasets['val'],
                      patience=10,
                      num_epochs=num_epochs,
                      learning_rate_scheduler=learning_rate_scheduler,
                      model_save_interval=10,
                      cuda_device=0)
    trainer.train()
    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])


    # In[17]:


    vocab.save_to_files("vocab-1908-reports-2015-01-16")


    # In[16]:


    # 12/18/2018 21:20:55 - INFO - allennlp.training.trainer -                Training |  Validation
    # 12/18/2018 21:20:55 - INFO - allennlp.training.trainer -   loss     |     0.023  |     0.072
    # 12/18/2018 21:20:55 - INFO - allennlp.training.trainer -   accuracy |     0.991  |     0.991
    # 12/18/2018 21:20:55 - INFO - allennlp.training.trainer -   Epoch duration: 00:00:42
    # 12/18/2018 21:20:55 - INFO - allennlp.training.trainer -   Estimated training time remaining: 0:22:18
    # 12/18/2018 21:20:55 - INFO - allennlp.training.trainer -   Epoch 19/49


    # In[20]:


    import datetime 
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')


    # out_model_name = f'BiLSTM-{timestamp}-epochs-{num_epochs}.pytorch'
    out_model_name = f'BiLSTM-{timestamp}-epochs-{num_epochs}-bs-{batch_size}-hiddim-{HIDDEN_DIM}-all-1908.pytorch'
    print(out_model_name)
    torch.save(model.state_dict(), out_model_name)


    from allennlp.nn import util
    model_state = torch.load(out_model_name, map_location=util.device_mapping(-1))
    # model.load_state_dict(model_state)



    from IPython.core.display import display, HTML
    from urllib.request import urlopen

    # color="255,127,80"  # coral
    color="coral"
    # print_color('sometext', color)


    ## Create and load css style

    colordict = {
     'findings': '#ff6666',
      'source': 'lightblue',
      'separator':'cyan',
      'see':'#ffcc66',
      'letter':'#ff99ff',
      'reference':'#6699ff',
      'notified': 'beige',
    }

    style = [".{label:12s} {{ background-color:{color};}}".format(label=label, color=color) for  label, color in colordict.items()]
    style = """<style>
    .title       {{ color:#939393; font-weight:800; float: left; width: 90px;}}
    .text        {{ float: left; width: 80%}}
    {}
    </style>
    """.format('\n'.join(style))
    print(style)
    with open('custom.css', 'w') as fh:
        fh.write(style)
        
    HTML(style)



    with open('special-cases-spacy.yaml') as fh:
        special_cases = yaml.load(fh)
        
    tokenizer_obj = SpacyTokenizer(language='en_core_web_sm', pos_tags=True)
    for case in special_cases:
        print(*case)
        tokenizer_obj.spacy.tokenizer.add_special_case(*case)

    tokenize = tokenizer_obj.split_words
    # tokenizer = lambda x: x.split(' ')

    # predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    predictor = AdvancedSentenceTaggerPredictor(model, dataset_reader=reader,
                                               tokenizer=tokenize)



    model.vocab._index_to_token['labels'].values()


    qry = 'A. Fine Needle Aspirate, Axilla: Scant benign fibroadipose tissue, see comment. B. Breast, Fine Needle Aspiration: Benign serous fluid and fat, see comment.'
    qry = qry.lower()#.replace('.',' ').replace(':','')

    qry =  re.sub(' +',' ', spu.normalize_punctuation(qry).lower().strip())

    tag_logits = predictor.predict(qry)['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)
    # print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])
    labels = [model.vocab.get_token_from_index(i, 'labels') for i in np.asarray(tag_logits).argmax(1)]

    txt = tokenize_sentence(predictor, qry)#.split(' ')
    # html_ = [get_color_html(kk,colordict[vv]) for kk,vv in zip(txt, labels)]
    html_ = [get_classy_html(kk, vv) for kk,vv in zip(txt, labels)]

    display(HTML(' '.join(html_)))



    # ## Predict

    # # sentence_indices
    # # set_onehot
    # token_index=pd.MultiIndex.from_tuples(list(zip(sentence_indices, token_indices)),names=['title', 'token_index'] )
    # df_logits = pd.DataFrame(set_logits,
    #              index=token_index,
    #              columns=model.vocab._index_to_token['labels'].values())
    # df_tokens = pd.Series(set_tokens, index=token_index, name='tokens')

    import sys




    labels = list(model.vocab._index_to_token['labels'].values())
    label_dict = {kk:vv for vv,kk in enumerate(labels)}


    # ## Label unlabelled data

    # In[51]:


    indices_labelled = [inst.index for inst in  datasets['all']]
    len(indices_labelled)


    # In[52]:


    get_ipython().system('ls data/ephi_*_4283_w_mined.csv')


    # In[53]:


    fn_4k = 'data/ephi_reports_4283_w_mined.csv'
    df_4k = pd.read_csv(fn_4k, index_col=0, squeeze=True)
    print(df_4k.shape[0])

    df_4k = df_4k.loc[~df_4k.index.isnull()]
    df_4k = df_4k[~df_4k.index.duplicated(keep='first')]
    print(df_4k.shape[0])

    df_4k_nolab = df_4k.drop(set(df_4k.index) & set(indices_labelled))
    print(df_4k_nolab.shape[0])


    # In[54]:


    fn_4k_norm = 'data/ephi_normalized_reports_4283_w_mined.csv'
    df_4k_norm = pd.read_csv(fn_4k_norm, index_col=0, squeeze=True)
    print(df_4k_norm.shape[0])

    df_4k_norm = df_4k_norm.loc[~df_4k_norm.index.isnull()]
    df_4k_norm = df_4k_norm[~df_4k_norm.index.duplicated(keep='first')]
    print(df_4k_norm.shape[0])

    df_4k_norm_nolab = df_4k_norm.drop(set(df_4k_norm.index) & set(indices_labelled))
    print(df_4k_norm_nolab.shape[0])

    df_4k_norm_nolab.map(lambda x: '  ' in x).sum()


    pd.Series(sum(df_4k_norm_nolab.str.split(' ')
        .map(lambda y: [x for x in y if '/' in x and not re.search('\d',x) and len(x)<6]),
        [])).value_counts()


    # In[59]:


    qry = df_4k_norm_nolab.iloc[0]
    qry


    # In[60]:


    dfres_logits = df_4k_norm_nolab.apply(lambda x: predictor.predict(x)['tag_logits'])
    dfres_logits = pd.concat({str(kk):vv for kk,vv in dfres_logits.map(pd.DataFrame).items()})
    dfres_logits.index.names = ['title', 'token_num']
    dfres_logits.columns=model.vocab._index_to_token['labels'].values()


    # In[123]:


    qry = 'A. Fine Needle Aspirate, Axilla: Scant benign fibroadipose tissue, see comment. B. Breast, Fine Needle Aspiration: Benign serous fluid and fat, see comment.'

    predictor.predict(qry)['tag_logits']


    dfres_tokens['labels'].astype(str).value_counts(normalize=True)*100



    dfres_prob = dfres_logits.applymap(lambda x: pd.np.exp(x)).div(
        dfres_logits.applymap(lambda x: pd.np.exp(x)).sum(1),
        axis=0)


    dfres_prob.head()



    with open('special-cases-spacy.yaml') as fh:
        special_cases = yaml.load(fh)

    # dfres_logits.loc['ZS05-7206']
    tokenizer_ = SpacyTokenizer(language='en_core_web_sm', pos_tags=True)
    for case in special_cases:
        print(*case)
        tokenizer_.spacy.tokenizer.add_special_case(*case)


    tokenizer = tokenizer_.split_words
    # tokenizer = lambda x: x.split(' ')


    x = df_4k_norm_nolab.iloc[0]
    # .map(lambda x: tokenize_sentence(x))
    # pd.Series(tokenize_sentence(x))



    dfres_tokens = tokenize_sentence_series(df_4k_norm_nolab)
    dfres_tokens['labels'] = dfres_logits.idxmax(1)
    dfres_tokens['entropy'] = (-np.log2(dfres_prob)*dfres_prob).sum(1)
    dfres_tokens.head()


    print(dfres_tokens.tokens.map(lambda x: '  ' in x).sum())


    dfres_tokens['labels'].astype(str).value_counts()

    dfres_tokens[dfres_tokens['labels'].isnull()]



    dfres_html = dfres_tokens.groupby('title').apply(lambda row: ' '.join([get_classy_html(to, la) for to,la in zip(row['tokens'], row['labels'])]))
    # display
    dfres_html[:5].reset_index().apply(lambda x: display(HTML('<div class="title">'+x['title']+ '</div>&emsp;&emsp;<div class="text">' + x[0] )),
                                                    axis=1)
    pass

    # list(zip(dfres_comb.loc['ZS05-7206','tokens'], (x.split(' '))))

    # list(zip(tokenizer.split_words(x), x.split(' ')))
    # len(tokenizer.split_words(x)), len(x.split(' '))
    # df_4k_norm_nolab.head()

    ds_json_segments = merge_annotation2json(dfres_tokens.tokens,
                                             df_4k_nolab)


    if 'entropy' not in ds_json_segments.columns:
        ds_json_segments = pd.concat([ds_json_segments, dfres_tokens['entropy'].groupby(level=0).max()], axis=1)
    ds_json_segments = ds_json_segments.sort_values('entropy', ascending=False)


    ds_json_segments.head()


    ds_json_segments['html'] = ds_json_segments.apply(
                        lambda x: '<div class="title">'+x['id'] + '</div>&emsp;&emsp;<div class="text">' +
                        annotate_html(x['text'], x['seq_annotations'],
                            start = 'start',
                            end = 'end',
                            label='label')+'</div>',
                                            axis=1)


    from parse_text_segmentation import parse_annotations
    from copy import deepcopy
    ds_json_segments_ = parse_annotations(deepcopy(ds_json_segments), handlabelled=False,
                          return_specimens=False)


    ds_json_segments_.segmented.isnull().sum()

    ds_json_segments_.entropy[ds_json_segments_.segmented.isnull()] = 3.0

    ds_json_segments_.sort_values('entropy', ascending=False, inplace=True)


    html_ = ds_json_segments['html'].iloc[0]
    html_
    display(HTML(html_))


    # In[93]:


    ds_json_segments['html'][15:30].map(lambda html_: display(HTML(html_)))


    # In[94]:


    ds_json_segments_[['entropy','text','seq_annotations']].head()


    get_ipython().system('rm data/ephi_reports_2899_annotated-BiLSTM-2019-01-10-21-10-epochs-30-bs-2-hiddim-6-all-1864.json')


    # In[96]:


    out_model_name = 'BiLSTM-2019-01-10-21-10-epochs-30-bs-2-hiddim-6-all-1864.pytorch'
    n_reports = ds_json_segments_.shape[0]
    fn_out = f'data/ephi_reports_{n_reports}_annotated-{out_model_name.strip(".pytorch")}.json'
    print(fn_out)
    ds_json_segments_[['id','entropy','text','seq_annotations']].to_json(fn_out, orient='records')


    # In[97]:


    fn_out_tokens = f'data/ephi_tokens_{n_reports}_annotated-{out_model_name.strip(".pytorch")}.csv'
    print(fn_out_tokens)
    dfres_tokens.to_csv(fn_out_tokens)


    # ### Issue: dash '-' handling: creates a shift 

    start = 0
    num = 10
    selection = ds_json_segments.entropy.sort_values(ascending=False)[start:start+num].index.tolist()


    # In[99]:


    ds_json_segments.loc[selection,'html'].map(lambda html_:display(HTML(html_)))
    pass


    # In[100]:


    dict_segments = ds_json_segments.sort_values('entropy', ascending=False)[['text', 'seq_annotations', 'id', 'entropy',]].to_dict( orient='records')

    df_relabelled = pd.read_json('data/ephi_reports_4283_w_mined_826_relabelled.json')
    df_relabelled.drop(['project', 'id'], axis=1, inplace=True)


    n_labelled = np.where(df_relabelled.seq_annotations.map(lambda x: any((y['manual'] for y in x))))[0][-1]
    df_relabelled = df_relabelled[:n_labelled]


