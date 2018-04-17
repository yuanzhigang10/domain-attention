"""
Data Loader for MDSD Dataset
"""
import os
import re
import numpy as np


# TODO: set your MDSD path here!
MDSD_PATH = os.path.expanduser('~/datasets/mdsd-v2')
assert os.path.exists(MDSD_PATH)

DOMAINS = ('books', 'dvd', 'electronics', 'kitchen')


def load_mdsd(domains, n_labeled=None):
    sorted_data_path = os.path.join(MDSD_PATH, 'sorted_data')  # .../mdsd-v2/sorted_data/
    print('loading data from {}'.format(sorted_data_path))
    texts = []
    s_labels = []
    d_labels = []
    sentiments = ('positive', 'negative')
    for d_id, d_name in enumerate(domains):
        for s_id, s_name in zip((1, 0, -1), sentiments):
            fpath = os.path.join(sorted_data_path, d_name, s_name + '.review')
            print(' - loading', d_name, s_name, end='')
            count = 0
            text = ''
            in_review_text = False
            with open(fpath, encoding='utf8', errors='ignore') as fr:
                for line in fr:
                    if '<review_text>' in line:
                        text = ''
                        in_review_text = True
                        continue
                    if '</review_text>' in line:
                        in_review_text = False
                        text = text.lower().replace('\n', ' ').strip()
                        text = re.sub(r'&[a-z]+;', '', text)
                        text = re.sub(r'\s+', ' ', text)
                        texts.append(text)
                        s_labels.append(s_id)
                        d_labels.append(d_id)
                        count += 1
                    if in_review_text:
                        text += line
                    # labeled cutoff
                    if (s_id >= 0) and n_labeled and (count == n_labeled):
                        break
            print(': %d texts' % count)
    print('data loaded')
    s_labels = np.asarray(s_labels, dtype='int')
    d_labels = np.asarray(d_labels, dtype='int')
    print(' - texts:', len(texts))
    print(' - s_labels:', len(s_labels))
    print(' - d_labels:', len(d_labels))

    return texts, s_labels, d_labels


if __name__ == '__main__':
    load_mdsd(domains=DOMAINS)
