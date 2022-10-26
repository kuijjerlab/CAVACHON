import numpy as np
import pandas as pd
import scipy.sparse
import tensorflow as tf
import unittest

from cavachon.utils.TensorUtils import TensorUtils
from sklearn.preprocessing import LabelEncoder

#%%
from cavachon.utils.TensorUtils import TensorUtils
df = pd.DataFrame({
    'A': [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
    'B': [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]
})
tensor, encoder_dict = TensorUtils.create_tensor_from_df(df, ['A', 'B'])

#%%
class TensorUtilsTestCase(unittest.TestCase):

  def test_create_tensor_from_df(self):
    df = pd.DataFrame({
      'A': [0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.],
      'B': [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]
    })
    tensor, encoder_dict = TensorUtils.create_tensor_from_df(df, ['A', 'B'])
    tensor_true = tf.convert_to_tensor([
        [1., 0.,  0.],
        [1., 0.,  1.],
        [1., 0.,  2.],
        [1., 0.,  3.],
        [1., 0.,  4.],
        [1., 0.,  5.],
        [0., 1.,  6.],
        [0., 1.,  7.],
        [0., 1.,  8.],
        [0., 1.,  9.],
        [0., 1., 10.],
        [0., 1., 11.]
    ])
    classes_true = np.array([0., 1.])
    
    self.assertTrue(encoder_dict['B'] is None, "'B' should not be one-hot encoded.")
    self.assertTrue(isinstance(encoder_dict['A'], LabelEncoder), "'A' should be one-hot encoded.")
    self.assertTrue(
        np.array_equal(classes_true, encoder_dict['A'].classes_),
        "The classes of the LabelEncoder is incorrect.")
    self.assertTrue(
        tf.reduce_all(tf.equal(tensor_true, tensor)),
        f"The created Tensor is not the same. Expect {tensor_true}, but get {tensor}."
    )

  def test_create_tensor_from_df_empty(self):
    df = pd.DataFrame({'A': [0., 0., 0., 0., 0]})
    tensor, encoder_dict = TensorUtils.create_tensor_from_df(df, ['C'])
    tensor_true = tf.convert_to_tensor([
        [0.],
        [0.],
        [0.],
        [0.],
        [0.]
    ])
    self.assertFalse('A' in encoder_dict, "'A' should not be one-hot encoded.")
    self.assertFalse('C' in encoder_dict, "'C' should not be one-hot encoded.")
    self.assertTrue(
        tf.reduce_all(tf.equal(tensor_true, tensor)),
        f"The created Tensor is not the same. Expect {tensor_true}, but get {tensor}."
    )

  def test_create_one_hot_encoded_tensor(self):
    data = pd.Series([0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.])
    tensor, encoder = TensorUtils.create_one_hot_encoded_tensor(data)

    tensor_true = tf.convert_to_tensor([
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [1., 0.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.],
        [0., 1.]
    ])
    classes_true = np.array([0., 1.])

    self.assertTrue(
        np.array_equal(classes_true, encoder.classes_),
        "The classes of the LabelEncoder is incorrect.")
    self.assertTrue(
        tf.reduce_all(tf.equal(tensor_true, tensor)),
        f"The created Tensor is not the same. Expect {tensor_true}, but get {tensor}."
    )
  
  def test_create_sparse_tensor(self):
    matrix = np.matrix([
        [0., 0., 1.],
        [0., 0., 0.],
        [1., 0., 0.]
    ])
    csr_matrix = scipy.sparse.csr_matrix(matrix)
    sparse_tensor = TensorUtils.spmatrix_to_sparse_tensor(csr_matrix)
    tensor = tf.sparse.to_dense(sparse_tensor)

    sparse_tensor_true = tf.SparseTensor(
        indices=[[0, 2], [2, 0]],
        values=[1., 1.],
        dense_shape=[3, 3]
    )
    tensor_true = tf.sparse.to_dense(sparse_tensor_true)
    self.assertTrue(
        tf.reduce_all(tf.equal(tensor_true, tensor)),
        f"The created Tensor is not the same. Expect {tensor_true}, but get {tensor}."
    )