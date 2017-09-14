#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#
import math
import numpy as np
import networkx as nx
from utils import LOGGER
from itertools import groupby


class TextRank(object):
    def __init__(self, max_iter=10000, alpha=0.85, tol=1e-6, using_matrix=True, word2vec=None, ndim=0):
        """
        Args:
            max_iter: maximum iterations for rank iteration
            alpha:
            tol:
            using_matrix: using matrix operation
        """
        self._max_iter = max_iter
        self._alpha = alpha
        self._tol = tol
        self._using_matrix = using_matrix

        self._word2vec = word2vec
        self._ndim = ndim
        self._unknown_vec = [0] * self._ndim

    @staticmethod
    def _right_stochastic_graph(graph):
        """reset the out-edge weight to: wji / sum_k wjk"""
        W = nx.DiGraph(graph)

        out_w_sum = {}  # out_wj = sum_k(wjk)
        for (u, _, w) in W.out_edges(data=True):
            out_w_sum[u] = out_w_sum.get(u, 0.0) + w['weight']

        # out_w_sum = W.out_degree()

        # normalize the edge weight
        for (u, v, w) in W.out_edges(data=True):
            if out_w_sum[u] > 0.0:
                w['weight'] = w.get('weight', 0.0) / out_w_sum[u]
            else:
                w['weight'] = 0.0

        return W

    def rank(self, sentences, theta=0.5):
        LOGGER.debug('build sentences similarity matrix')
        n_sentences = len(sentences)

        graph = np.zeros((n_sentences, n_sentences))  # adj-matrix

        for i in xrange(n_sentences):
            for j in xrange(i+1, n_sentences):
                weight = self.sim_word_embedding(sentences[i][1], sentences[j][1])

                if weight >= theta:
                    graph[i, j] = weight
                    graph[j, i] = weight
        nx_graph = nx.from_numpy_matrix(graph)

        D = nx_graph
        if not D.is_directed():
            D = D.to_directed()

        W = self._right_stochastic_graph(D)
        N = W.number_of_nodes()

        # power iteration
        # x = (1-d) + d * x' * w
        x = dict.fromkeys(W, 1.0 / N)
        if self._using_matrix:
            x = x.values()
            w = np.zeros((N, N))
            for (u, v, _w) in W.out_edges(data=True):
                w[u][v] = _w['weight']

            for i in xrange(self._max_iter):
                x_last = x
                x = 1 - self._alpha + self._alpha * np.matmul(x_last, w)

                delta = x - x_last

                err = np.linalg.norm(delta)
                LOGGER.error('iter: %d, err: %.5f' % (i, err))
                if err < N * self._tol:
                    return sorted(
                        [(x[n], sentences[n][0]) for n in xrange(len(x))], key=lambda v: v[0], reverse=True
                    )
        else:
            for i in xrange(self._max_iter):
                x_last = x

                x = dict.fromkeys(x_last.keys(), 0)
                for n in x:
                    sum_in_nbr = sum([w['weight'] * x_last.get(u, 0.0) for (u, _, w) in W.in_edges(n, data=True)])
                    x[n] = 1 - self._alpha + self._alpha * sum_in_nbr

                # check convergence
                err = sum([abs(x[n] - x_last[n]) for n in x])
                LOGGER.error('iter: %d, err: %.5f' % (i, err))
                if err < N * self._tol:
                    return sorted(
                        [(r, sentences[n][0]) for n, r in x.items()], key=lambda v: v[0], reverse=True
                    )

        raise nx.NetworkXError('text-rank: power iteration failed to converge in %d iterations', self._max_iter)

    @staticmethod
    def cosine(v1, v2):
        assert isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray), 'numpy 1-d array is required'
        assert v1.shape == v2.shape, 'size should be equal'

        denominator = np.linalg.norm(v1) * np.linalg.norm(v2)

        sim = 1e-6
        if denominator > 0.0:
            numerator = np.matmul(v1, v2)
            sim = numerator / denominator

        return sim

    @staticmethod
    def sim_jaccard(s1, s2):
        assert isinstance(s1, list) and isinstance(s2, list), 'list is required for sentences'

        numerator = len(set(s1).intersection(set(s2))) * 1.0
        denominator = math.log(float(len(s1))) + math.log(float(len(s1)))
        if abs(denominator) <= 1e-10:
            return 0.0
        else:
            return numerator / denominator

    def sim_semantic_jaccard(self, s1, s2, alpha=0.6):
        assert isinstance(s1, list) and isinstance(s2, list), 'list is required for sentences'
        assert self._word2vec, 'word vec is required'

        n1 = len(s1)
        n2 = len(s2)

        if n1 < n2:
            s2, s1 = s1, s2
            n2, n1 = n1, n2

        LOGGER.debug('build word similar matrix')
        m = np.zeros((n1, n2))
        for i in xrange(n1):
            for j in xrange(n2):
                vs1i = np.array(self._word2vec.get(s1[i], self._unknown_vec))
                vs2j = np.array(self._word2vec.get(s2[j], self._unknown_vec))

                m[i, j] = self.cosine(vs1i, vs2j)

        LOGGER.debug('calculate similarity')
        numerator = 0.0
        while True and m.size > 0:
            max_m_i, max_m_j = np.unravel_index(m.argmax(), m.shape)
            max_m = m[max_m_i, max_m_j]

            if max_m < alpha:
                break

            numerator += max_m
            n_row, n_col = m.shape
            row = np.reshape(range(0, max_m_i) + range(max_m_i + 1, n_row), (-1, 1))
            col = range(0, max_m_j) + range(max_m_j + 1, n_col)

            if len(row) > 0 and len(col) > 0:
                m = m[row, col]
            else:
                m = np.array([[]])

        beta = m.size
        m_diff = (1 - m).sum()

        denominator = numerator + beta * m_diff
        if denominator > 0:
            return numerator / denominator
        else:
            return 1e-6

    def sim_word_embedding(self, s1, s2):
        vs1 = np.zeros(self._ndim)
        vs2 = np.zeros(self._ndim)
        for w in s1:
            vs1 += np.array(self._word2vec.get(w, self._unknown_vec))

        vs1 /= len(s1)

        for w in s2:
            vs2 += np.array(self._word2vec.get(w, self._unknown_vec))

        vs2 /= len(s2)

        return self.cosine(vs1, vs2)


def _load_word2vec(fname, dim=256):
    length = dim + 1

    word2vec = {}

    with open(fname, 'rb') as fp:
        counter = 0
        for line in fp:
            if counter % 1000 == 0:
                LOGGER.debug('loading vector: %d' % counter)

            line = line.strip().split()
            if len(line) == length:
                word2vec[line[0]] = [float(v) for v in line[1:]]

            counter += 1

    return word2vec


def _load_articles(fname):
    articles = []
    with open(fname, 'rb') as fp:
        counter = 0
        for line in fp:
            if counter % 1000 == 0:
                LOGGER.debug('loading article: %d' % counter)

            counter += 1

            line = line.strip().split('\001')
            doc_id = line[0]
            doc_url = line[1]
            sen_id = line[3]
            sen = line[4]
            words = [w.strip() for w in line[5].split("(")[-1].split(")")[0].split(",")]

            articles.append([doc_id, doc_url, sen_id, sen, words])

    articles = sorted(articles, key=lambda a: a[0])

    # articles:
    # {
    #   doc_id: {
    #       url: ''.
    #       sen: {
    #           sen_id: sen
    #       },
    #       words: [
    #           (sen_id, words)
    #       ]
    #   }
    # }

    grouped_articles = {}
    for (doc_id, doc_url), group in groupby(articles, lambda a: (a[0], a[1])):
        sen = {g[2]: (g[3], g[4]) for g in group}
        words = [(sen_id, _words) for sen_id, (_, _words) in sen.items()]

        cur_article = {
            'url': doc_url,
            'sen': sen,
            'words': words
        }

        grouped_articles[doc_id] = cur_article

    return grouped_articles


if __name__ == '__main__':
    fname_texts = './data/articles.txt'
    fname_word2vec = './data/vectors.20170613.w2v'
    dim = 256

    articles = _load_articles(fname_texts)

    word2vec = _load_word2vec(fname_word2vec, dim=dim)
    tr = TextRank(word2vec=word2vec, ndim=dim)

    for doc_id, article in articles.items():
        words = article['words']
        for r, _id in tr.rank(words):
            print '{0}\t{1}\t{2}\t{3}'.format(doc_id, article['url'], r, article['sen'].get(_id)[0])
