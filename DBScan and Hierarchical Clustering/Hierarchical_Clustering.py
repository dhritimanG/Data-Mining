from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import numpy as np
from __future__ import division

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 20:09:49 2018

@author: dhritiman
"""

import os
import functools
import operator
import gzip
import struct
import array
import tempfile
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve  # py2
try:
    from urllib.parse import urljoin
except ImportError:
    from urlparse import urljoin
import numpy as np


# the url can be changed by the users of the library (not a constant)
datasets_url = 'http://yann.lecun.com/exdb/mnist/'


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def download_file(fname, target_dir=None, force=False):
    """Download fname from the datasets_url, and save it to target_dir,
    unless the file already exists, and force is False.

    Parameters
    ----------
    fname : str
        Name of the file to download

    target_dir : str
        Directory where to store the file

    force : bool
        Force downloading the file, if it already exists

    Returns
    -------
    fname : str
        Full path of the downloaded file
    """
    if not target_dir:
        target_dir = tempfile.gettempdir()
    target_fname = os.path.join(target_dir, fname)

    if force or not os.path.isfile(target_fname):
        url = urljoin(datasets_url, fname)
        urlretrieve(url, target_fname)

    return target_fname


def parse_idx(fd):
    """Parse an IDX file, and return it as a numpy array.

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse

    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file

    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


def download_and_parse_mnist_file(fname, target_dir=None, force=False):
    """Download the IDX file named fname from the URL specified in dataset_url
    and return it as a numpy array.

    Parameters
    ----------
    fname : str
        File name to download and parse

    target_dir : str
        Directory where to store the file

    force : bool
        Force downloading the file, if it already exists

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    """
    fname = download_file(fname, target_dir=target_dir, force=force)
    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        return parse_idx(fd)


def train_images():
    """Return train images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    train_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('train-images-idx3-ubyte.gz')


def test_images():
    """Return test images from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    test_images : numpy.ndarray
        Numpy array with the images in the train MNIST database. The first
        dimension indexes each sample, while the other two index rows and
        columns of the image
    """
    return download_and_parse_mnist_file('t10k-images-idx3-ubyte.gz')


def train_labels():
    """Return train labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    train_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('train-labels-idx1-ubyte.gz')


def test_labels():
    """Return test labels from Yann LeCun MNIST database as a numpy array.
    Download the file, if not already found in the temporary directory of
    the system.

    Returns
    -------
    test_labels : numpy.ndarray
        Numpy array with the labels 0 to 9 in the train MNIST database.
    """
    return download_and_parse_mnist_file('t10k-labels-idx1-ubyte.gz')



# Hierarchical_Clustering  using sklearn package
class Hierarchical_Clustering:
    def __init__(self, n_clusters, affinity, linkage):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.pred = None
        self.centroids_dict = None
        self.n = None
        
    def fit(self, dataset, labels):
        self.n = dataset.shape[0]
        Hclustering = AgglomerativeClustering(n_clusters = self.n_clusters,
                                              affinity = self.affinity,
                                              linkage = self.linkage)
        
        # Fit the model on the dataset
        Hclustering.fit(dataset)
        self.pred = Hclustering.labels_
        self.centroids_dict = self.form_clusters(self.pred)
        
        # Test purity of the model
        self.compute_purity_gini(self.centroids_dict, labels, self.n, self.n_clusters)
        
    # Generate the centroids from the Hierarchical_Clustering clustering
    def form_clusters(self, labels):
        centroids_dict = dict()
        for i in range(len(labels)):
            if labels[i] in centroids_dict:
                centroids_dict[labels[i]].append(i)
            else:
                centroids_dict[labels[i]] = [i]
        
        return centroids_dict
        
    # Function to calculate the purity and gini-index
    def compute_purity_gini(self, centroids_dict, labels, n, n_clusters):
        purity_cnt = 0
        gini = 0
        for i in centroids_dict:
            k = list(centroids_dict[i])
            y = []
            for i in range(len(k)):
                y.append(labels[k[i]])
            counts_dict = dict(Counter(y))
            tmp = max(counts_dict, key = counts_dict.get)
            purity_cnt += counts_dict[tmp]
        
            values = np.array(list(counts_dict.values()))
            values = np.square(values/np.sum(values))
            gini += 1 - np.sum(values)
            
        purity = purity_cnt/n
        gini_coef = gini/n_clusters
        print("Purity:", purity, "Gini:", gini_coef)
        

images_train = train_images()
images_test = test_images()

# Converting to 2-d array
x_train = images_train.reshape((images_train.shape[0], images_train.shape[1] * images_train.shape[2]))
x_test = images_test.reshape((images_test.shape[0], images_test.shape[1] * images_test.shape[2]))
train_labels = train_labels()
test_labels = test_labels()

# Normalize the datasets
x_train = x_train/255
x_test = x_test/255
        

# Train the model with different parameters
model = Hierarchical_Clustering(n_clusters = 10, affinity = 'cosine', linkage = 'average')
model.fit(x_test, test_labels)
        
model = Hierarchical_Clustering(n_clusters = 10, affinity = 'euclidean', linkage = 'ward')
model.fit(x_test, test_labels)      

model = Hierarchical_Clustering(n_clusters = 20, affinity = 'euclidean', linkage = 'ward')
model.fit(x_test, test_labels)

model = Hierarchical_Clustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
model.fit(x_test, test_labels)

model = Hierarchical_Clustering(n_clusters = 40, affinity = 'euclidean', linkage = 'ward')
model.fit(x_test, test_labels)
