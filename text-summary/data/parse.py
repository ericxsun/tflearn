#!/user/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright @2017 Research, NTC Inc. (ntc.com)
#
# Author: Eric x.sun <eric.x.sun@gmail.com>
#

import codecs
from optparse import OptionParser

import re
from znlp.my_replacer import MyReplacer
import sys

import unicodedata

LOGGER = codecs.getwriter('utf-8')(sys.stderr)

SOURCE_ENCODING = 'gb18030'

DOC_START = '<doc>'
DOC_END = '</doc>'
DOC_URL_L = '<url>'
DOC_URL_R = '</url>'
DOC_NUM_L = '<docno>'
DOC_NUM_R = '</docno>'
DOC_TITLE_L = '<contenttitle>'
DOC_TITLE_R = '</contenttitle>'
DOC_CONTENT_L = '<content>'
DOC_CONTENT_R = '</content>'


def normalize(content):
    r_invalid_chars = {
        re.compile(u'\ue40c', re.UNICODE): '\n'
    }

    content = unicodedata.normalize('NFKC', content)
    content = MyReplacer.replace_non_printable_character(content)

    for key, value in r_invalid_chars.items():
        content = MyReplacer.replace_with_pattern(key, content, replacement=value)

    return content


def parse(data, out):
    fin = codecs.open(data, 'rb', encoding=SOURCE_ENCODING)
    if out:
        fout = codecs.open(out, 'wb', encoding='utf-8')
    else:
        fout = codecs.getwriter('utf-8')(sys.stdout)

    counter = 0
    cur_article = []
    for obj in fin:
        counter += 1
        if counter % 1000 == 0:
            LOGGER.write('process line: %d\n' % counter)

        obj = obj.strip()
        if obj.startswith(DOC_URL_L):
            cur_article.append(normalize(obj.lstrip(DOC_URL_L).rstrip(DOC_URL_R)))
        elif obj.startswith(DOC_NUM_L):
            cur_article.append(normalize(obj.lstrip(DOC_NUM_L).rstrip(DOC_NUM_R)))
        elif obj.startswith(DOC_TITLE_L):
            cur_article.append(normalize(obj.lstrip(DOC_TITLE_L).rstrip(DOC_TITLE_R)))
        elif obj.startswith(DOC_CONTENT_L):
            cur_article.append(normalize(obj.lstrip(DOC_CONTENT_L).rstrip(DOC_CONTENT_R)))
        elif obj.startswith(DOC_END):
            if cur_article:
                fout.write(u'{0}\n'.format(u'\001'.join(cur_article)))
            cur_article = []


if __name__ == '__main__':
    parser = OptionParser(usage="""%prog -c cmd -d data -o out""")

    parser.add_option(
        '-d', '--data',
        help=u'(Required) The file name of data.'
    )

    parser.add_option(
        '-o', '--out',
        help=u'(Optional) The file name of output.'
    )

    if not sys.argv[1:]:
        parser.print_help()
        exit(1)

    (opts, args) = parser.parse_args()

    main(data=opts.data, out=opts.out)
