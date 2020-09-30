#!/usr/bin/env python
# coding: utf-8

# allennlp v.0.9.0

from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np
import re
from allennlp.data import Instance
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor

torch.manual_seed(1)
import pandas as pd
from allennlp.data.dataset_readers import DatasetReader
from segment_allennlp import (PosDatasetReader, LstmTagger,
                              AdvancedBucketIterator,
                              permute_token, AdvancedSentenceTaggerPredictor)

import segm_prep_utils as spu
from functools import reduce
from functools import partial
from collections import Counter

# example of an input file:
# id1243 | Patient#%#bg John#%#name Smith#%#name

reader = PosDatasetReader(index_sep='|', word_tag_delimiter='#%#')
fns = {}
datasets = {}
for set_ in ['train', 'val', 'test', 'regexed']:
    fns[set_] = "data/ephi_reports-571-entries-2018-12-11-formatted-noacc-{}.csv".format(set_)
#     fns[set_] = 'data/ephi_reports-1864-entries-2018-12-27-formatted-{}.csv'.format(set_)
    datasets[set_] = reader.read(fns[set_],)
    print(set_, list(reduce(lambda x,y: x|y, (set(inst.fields['labels'])  for inst in  datasets[set_] ))), sep='\t')

def mutate(inst, frequency=0.5):
    words,  labels = list(zip(*[(word.text, label) for word, label in
                                zip(inst.fields['sentence'], inst.fields['labels'].labels, )
                                if word.text not in PUNCTUATION]))

    words = list(words)
    if np.random.rand()>frequency:
        words[np.random.randint(len(words))] = 'unk'
    return text_to_instance(words,  labels)

set_ = 'train'
fns[set_] = "data/ephi_reports-571-entries-2018-12-11-formatted-noacc-{}.csv".format(set_)
PUNCTUATION = set(list('.,:;()'))
reader.remove_tokens = PUNCTUATION
datasets[set_ + '_no_punctuation'] = reader.read(fns[set_],)
reader.remove_tokens = []


vocab = Vocabulary.from_instances(datasets['train'] + datasets['val'] 
#                                   + train_dataset_mutated
                            )

labels = [vocab.get_token_from_index(i, 'labels') for i in range(0, vocab.get_vocab_size('labels'))]
labels

label_counts = reduce(lambda x,y: x + y, (Counter(inst.fields['labels'])  for inst in  datasets['train'] ))
label_counts = pd.Series(label_counts)


from torch.optim.lr_scheduler import MultiStepLR
from allennlp.training.learning_rate_schedulers.learning_rate_scheduler import _PyTorchLearningRateSchedulerWrapper

# WRAPPED FOR HYPER-PARAMETER OPTIMISATION
# serialization_dir = 'checkpoints'
def objective_kw(num_epochs = 10, lr=0.1, lr_gamma=0.25,
                EMBEDDING_DIM = 16,
                HIDDEN_DIM = 6,
                DROPOUT = 0.5,
                AUGMENT = True,
                weight_exponent = 1.0,
             ):

    weights = label_counts.map(lambda x: x**(-1/(1+weight_exponent))).loc[labels]
    weights = weights /((label_counts*weights).mean() /label_counts.mean())

    loss_params = dict(alpha=weights.values,
                       gamma=None)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM,
                                               batch_first=True,
                                               bidirectional=True,
                                               dropout=DROPOUT))

    model = LstmTagger(word_embeddings, lstm, vocab, loss_params=loss_params)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    iterator = BucketIterator(batch_size=2, sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)

    if AUGMENT:
        iterator = AdvancedBucketIterator(batch_size=2, 
                                          sorting_keys=[("sentence", "num_tokens")],
                                          preprocess=partial(permute_token, frequency=0.2),
                                         )
        iterator.index_with(vocab)

        val_iterator = AdvancedBucketIterator(batch_size=2, 
                                          sorting_keys=[("sentence", "num_tokens")],
                                         )
        val_iterator.index_with(vocab)
    else:
        val_iterator = iterator

    for _ in range(1):
        optimizer = optim.SGD(model.parameters(), lr=lr)

        learning_rate_scheduler = _PyTorchLearningRateSchedulerWrapper(MultiStepLR(optimizer,
                                                                                [10, 20, 40],
                                                                                gamma=lr_gamma, last_epoch=-1))


        trainer = Trainer(model=model,
                          optimizer=optimizer,
                          iterator=iterator,
                          validation_iterator = val_iterator,
                          train_dataset=datasets['train_no_punctuation'] + datasets['train'],
                          validation_dataset=datasets['val'],
                          patience=10,
                          num_epochs=num_epochs,
                          learning_rate_scheduler=learning_rate_scheduler,
#                           model_save_interval=10,
    #                       serialization_dir=serialization_dir,
    #                       num_serialized_models_to_keep=10,
                         )
        res = trainer.train()
        return 1 - res['validation_accuracy'] # res['validation_loss']

def objective(args):
    return objective_kw(**args)

# minimize the objective over the space
from hyperopt import hp, fmin, tpe, space_eval, Trials, STATUS_OK
import pickle
import json

# Initialize an empty trials database
trials = Trials()


space = dict(
#     num_epochs = 3,
    weight_exponent = hp.uniform('weight_exponent', 0.1, 1.0),
    lr=hp.loguniform('lr', 1e-3, 5e-1), 
    lr_gamma= hp.uniform('lr_gamma', 0.1, 0.5),
#     EMBEDDING_DIM = 16,
#     HIDDEN_DIM = 6,
    HIDDEN_DIM = hp.choice('HIDDEN_DIM', [3, 6, 9]),
    #DROPOUT = hp.uniform('DROPOUT', 0.2, 0.8),
    AUGMENT = hp.choice('AUGMENT', [True, False]),
)


best = fmin(objective, space, 
            algo=tpe.suggest, max_evals=100,
            trials=trials,)


print('-'*40)
print(best)

best_file = 'hyperopt-weighted_loss-r1.json'
with open(trials_file, 'w') as fh:
    json.dump(trials, fh)

trials_file = 'hyperopt-weighted_loss-r1-trials.pickle'

# The trials database now contains 100 entries, it can be saved/reloaded with pickle or another method
try:
    with open(trials_file, 'w') as fh:
        json.dump(trials, fh)
except:
    for trial in trials.trials:
        if 'result' in trials.keys():
            # https://stackoverflow.com/questions/15411107/delete-a-dictionary-item-if-the-key-exists
            trials['result'].pop('model', None)
    # proceed with pickling
    with open(trials_file, 'w') as fh:
        json.dump(trials, fh)

