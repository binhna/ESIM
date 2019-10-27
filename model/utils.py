import torch
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, Vectors
import spacy
import os
import random
SEED = 1234
# from nltk import word_tokenize

def tokenizer(text):
    spacy_en = spacy.load('en')
    fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    trans_map = str.maketrans(fileters, " " * len(fileters))
    text = text.translate(trans_map)
    text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']

    tokenized_text = []
    auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', "'s"]
    for token in text:
        if token == "n't":
            tmp = 'not'
        elif token == "'ll":
            tmp = 'will'
        elif token in auxiliary_verbs:
            tmp = 'be'
        else:
            tmp = token
        tokenized_text.append(tmp)
    return tokenized_text

class SNLI():
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, tokenize='spacy', lower=True, include_lengths=True)
        # self.LABEL = data.Field(sequential=False, unk_token=None)
        self.LABEL = data.LabelField()
        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL,
                                                               root='/media/fch/Data/leo/text-similarity/data')
        vectors = Vectors(name='/media/fch/Data/leo/text-similarity/glove/glove.840B.300d.txt',
                          cache='/media/fch/Data/leo/text-similarity/.vector_cache')
        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=vectors)
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.premise), len(x.hypothesis))
        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=args.device,
                                       sort_key=sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])


class Quora():
    def __init__(self, args):
        self.RAW = data.RawField(is_target=False)
        self.TEXT = data.Field(batch_first=True, tokenize='spacy', lower=True)
        # self.LABEL = data.Field(sequential=False, unk_token=None)
        self.LABEL = data.LabelField()

        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='/media/fch/Data/leo/text-similarity/data/quora',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('label', self.LABEL),
                    ('q1', self.TEXT),
                    ('q2', self.TEXT),
                    ('id', self.RAW)])
        vectors = Vectors(name='/media/fch/Data/leo/text-similarity/glove/glove.840B.300d.txt',
                          cache='/media/fch/Data/leo/text-similarity/.vector_cache')
        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=vectors)
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits((self.train, self.dev, self.test),
                                                                                    batch_sizes=[args.batch_size] * 3,
                                                                                    device=args.device,
                                                                                    sort_key=sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])


class WIKI():
    def __init__(self, args):
        self.TEXT = data.Field(lower=args.lower, tokenize=lambda x: x.split())
        # We'll use NestedField to tokenize each word into list of chars
        if args.char:
            CHAR_NESTING = data.Field()
            self.char_text = data.NestedField(CHAR_NESTING)
        self.LABEL = data.LabelField()
        self.ids = data.Field(sequential=True, use_vocab=True)
        if args.char:
            fields = {'question': ('question', self.TEXT), 'char_question': ('question_c', self.char_text),
                      'label': ('label', self.LABEL), 'text': ('answer', self.TEXT), 'char_text': ('answer_c', self.char_text)}
            test_fields = {'__id__': ('q_id', self.ids), 'question': (
                'question', self.TEXT), 'char_question': ('question_c', self.char_text),
                'text': ('answer', self.TEXT), 'char_text': ('answer_c', self.char_text),
                'id': ('a_id', self.ids)}
        else:
            fields = {'question': ('question', self.TEXT),
                      'label': ('label', self.LABEL), 'text': ('answer', self.TEXT)}
            test_fields = {'__id__': ('q_id', self.ids), 'id': ('a_id', self.ids),
                           'question': ('question', self.TEXT), 'text': ('answer', self.TEXT)}

        data_zalo = data.TabularDataset(path='../wikiqa_zalo/data/train_pr.json',
                                        format='json',
                                        fields=fields)
        data_submission = data.TabularDataset(path='../wikiqa_zalo/data/test_pr_submission.json',
                                              format='json',
                                              fields=test_fields)

        self.train, self.test = data_zalo.split(
            0.8, random_state=random.seed(SEED))
        self.train, self.dev = self.train.split(
            0.8, random_state=random.seed(SEED))
        # len(data_zalo), len(train), len(test), len(valid)

        self.TEXT.build_vocab(self.train, self.dev, self.test)
        self.ids.build_vocab(data_submission)
        if args.char:
            self.char_text.build_vocab(self.train, self.dev, self.test)

        if args.word_vectors:
            if os.path.isfile(args.vector_cache):
                print('Found pretrained word embeddings')
                cache, name = '/'.join(args.vector_cache.split('/')
                                       [:-1]), args.vector_cache.split('/')[-1]
                vectors = Vectors(cache=cache, name=name)
                self.TEXT.vocab.set_vectors(
                    vectors.stoi, vectors.vectors, vectors.dim)
                # inputs.vocab.vectors = torch.load(args.vector_cache)
            else:
                print('Not found pretrained word embeddings\nDownloading')
                self.TEXT.vocab.load_vectors(args.word_vectors)
                makedirs(os.path.dirname(args.vector_cache))
                torch.save(self.TEXT.vocab.vectors, args.vector_cache)

        if args.char_vectors and args.char:
            print('Found pretrained character embeddings')
            cache, name = '/'.join(args.char_vectors.split('/')
                                   [:-1]), args.char_vectors.split('/')[-1]
            char_vectors = Vectors(cache=cache, name=name)
            self.char_text.vocab.set_vectors(
                char_vectors.stoi, char_vectors.vectors, char_vectors.dim)
        self.LABEL.build_vocab(self.train)

        def sort_key(x): return data.interleave_keys(
            len(x.question), len(x.answer))
        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits(
            (self.train, self.dev,
             self.test), batch_size=args.batch_size, device=args.device,
            sort_key=sort_key,
            sort_within_batch=False)
        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])

        self.submission_iter = data.BucketIterator(
            data_submission,
            batch_size=args.batch_size,
            device=args.device,
            sort_within_batch=False)
