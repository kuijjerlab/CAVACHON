import anndata
import numpy as np
import pandas as pd
import unittest

from cavachon.utils.AnnDataUtils import AnnDataUtils
from scipy.sparse import csr_matrix

class AnnDataUtilsTestCase(unittest.TestCase):

  def test_reorder_adata_dict(self):
    obs_1 = pd.DataFrame({'Obs': ['S-A', 'S-B', 'S-C']}, index=pd.Index(['A', 'B', 'C']))  
    var_1 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_1 = csr_matrix(np.matrix([[0., 1.], [2., 3.], [4., 5.]]))
    adata_1 = anndata.AnnData(X=mat_1, obs=obs_1, var=var_1)

    obs_2 = pd.DataFrame({'Obs': ['S-C', 'S-A', 'S-B']}, index=pd.Index(['C', 'A', 'B']))  
    var_2 = pd.DataFrame({'Var': ['A-1', 'A-2']}, index=pd.Index(['1', '2']))
    mat_2 = csr_matrix([[14., 15.], [10., 11.], [12., 13.]])
    adata_2 = anndata.AnnData(X=mat_2, obs=obs_2, var=var_2)

    obs_3 = pd.DataFrame({'Obs': ['S-A', 'S-B', 'S-C']}, index=pd.Index(['A', 'B', 'C']))  
    var_3 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_3 = csr_matrix(np.matrix([[0., 1.], [2., 3.], [4., 5.]]))
    adata_3 = anndata.AnnData(X=mat_3, obs=obs_3, var=var_3)

    obs_4 = pd.DataFrame({'Obs': ['S-A', 'S-B', 'S-C']}, index=pd.Index(['A', 'B', 'C']))  
    var_4 = pd.DataFrame({'Var': ['A-1', 'A-2']}, index=pd.Index(['1', '2']))
    mat_4 = csr_matrix(np.matrix([[10., 11.], [12., 13.], [14., 15.]]))
    adata_4 = anndata.AnnData(X=mat_4, obs=obs_4, var=var_4)

    unordered_adata_dict = {'mod_1': adata_1, 'mod_2': adata_2}
    ordered_adata_dict = {'mod_1': adata_3, 'mod_2': adata_4}
    result_adata_dict = AnnDataUtils.reorder_adata_dict(unordered_adata_dict)

    for mod in ordered_adata_dict.keys():
      self.assertTrue(
        np.array_equal(
            ordered_adata_dict[mod].X.toarray(), result_adata_dict[mod].X.toarray()), 
        f"The matrix of modality {mod} is not sorted correctly.")
      for annot in ['var', 'obs']:
        self.assertTrue(
            getattr(ordered_adata_dict[mod], annot).equals(
              getattr(result_adata_dict[mod], annot)),
            f"The {annot} dataframe of modality {mod} is not sorted correctly.")
  
  def test_reorder_adata_dict_missing_index(self):
    obs_1 = pd.DataFrame({'Obs': ['S-A', 'S-B', 'S-C']}, index=pd.Index(['A', 'B', 'C']))  
    var_1 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_1 = csr_matrix(np.matrix([[0., 1.], [2., 3.], [4., 5.]]))
    adata_1 = anndata.AnnData(X=mat_1, obs=obs_1, var=var_1)

    obs_2 = pd.DataFrame({'Obs': ['S-C', 'S-A']}, index=pd.Index(['C', 'A']))  
    var_2 = pd.DataFrame({'Var': ['A-1', 'A-2']}, index=pd.Index(['1', '2']))
    mat_2 = csr_matrix([[14., 15.], [10., 11.]])
    adata_2 = anndata.AnnData(X=mat_2, obs=obs_2, var=var_2)

    obs_3 = pd.DataFrame({'Obs': ['S-A', 'S-C']}, index=pd.Index(['A', 'C']))  
    var_3 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_3 = csr_matrix(np.matrix([[0., 1.], [4., 5.]]))
    adata_3 = anndata.AnnData(X=mat_3, obs=obs_3, var=var_3)

    obs_4 = pd.DataFrame({'Obs': ['S-A', 'S-C']}, index=pd.Index(['A', 'C']))  
    var_4 = pd.DataFrame({'Var': ['A-1', 'A-2']}, index=pd.Index(['1', '2']))
    mat_4 = csr_matrix(np.matrix([[10., 11.], [14., 15.]]))
    adata_4 = anndata.AnnData(X=mat_4, obs=obs_4, var=var_4)

    unordered_adata_dict = {'mod_1': adata_1, 'mod_2': adata_2}
    ordered_adata_dict = {'mod_1': adata_3, 'mod_2': adata_4}
    result_adata_dict = AnnDataUtils.reorder_adata_dict(unordered_adata_dict)

    for mod in ordered_adata_dict.keys():
      self.assertTrue(
        np.array_equal(
            ordered_adata_dict[mod].X.toarray(), result_adata_dict[mod].X.toarray()), 
        f"The matrix of modality {mod} is not sorted correctly.")
      for annot in ['var', 'obs']:
        self.assertTrue(
            getattr(ordered_adata_dict[mod], annot).equals(
              getattr(result_adata_dict[mod], annot)),
            f"The {annot} dataframe of modality {mod} is not sorted correctly.")
  
  def test_reorder_adata_dict_provide_index(self):
    obs_1 = pd.DataFrame({'Obs': ['S-A', 'S-B', 'S-C']}, index=pd.Index(['A', 'B', 'C']))  
    var_1 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_1 = csr_matrix(np.matrix([[0., 1.], [2., 3.], [4., 5.]]))
    adata_1 = anndata.AnnData(X=mat_1, obs=obs_1, var=var_1)

    obs_2 = pd.DataFrame({'Obs': ['S-C', 'S-A', 'S-B']}, index=pd.Index(['C', 'A', 'B']))  
    var_2 = pd.DataFrame({'Var': ['A-1', 'A-2']}, index=pd.Index(['1', '2']))
    mat_2 = csr_matrix([[14., 15.], [10., 11.], [12., 13.]])
    adata_2 = anndata.AnnData(X=mat_2, obs=obs_2, var=var_2)

    obs_3 = pd.DataFrame({'Obs': ['S-B', 'S-C', 'S-A']}, index=pd.Index(['B', 'C', 'A']))  
    var_3 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_3 = csr_matrix(np.matrix([[2., 3.], [4., 5.], [0., 1.]]))
    adata_3 = anndata.AnnData(X=mat_3, obs=obs_3, var=var_3)

    obs_4 = pd.DataFrame({'Obs': ['S-B', 'S-C', 'S-A']}, index=pd.Index(['B', 'C', 'A']))  
    var_4 = pd.DataFrame({'Var': ['A-1', 'A-2']}, index=pd.Index(['1', '2']))
    mat_4 = csr_matrix(np.matrix([[12., 13.], [14., 15.], [10., 11.]]))
    adata_4 = anndata.AnnData(X=mat_4, obs=obs_4, var=var_4)

    unordered_adata_dict = {'mod_1': adata_1, 'mod_2': adata_2}
    ordered_adata_dict = {'mod_1': adata_3, 'mod_2': adata_4}
    result_adata_dict = AnnDataUtils.reorder_adata_dict(
        unordered_adata_dict, pd.Index(['B', 'C', 'A']))

    for mod in ordered_adata_dict.keys():
      self.assertTrue(
        np.array_equal(
            ordered_adata_dict[mod].X.toarray(), result_adata_dict[mod].X.toarray()), 
        f"The matrix of modality {mod} is not sorted correctly.")
      for annot in ['var', 'obs']:
        self.assertTrue(
            getattr(ordered_adata_dict[mod], annot).equals(
              getattr(result_adata_dict[mod], annot)),
            f"The {annot} dataframe of modality {mod} is not sorted correctly.")

  def test_reorder_adata(self):
    obs_1 = pd.DataFrame({'Obs': ['S-A', 'S-B', 'S-C']}, index=pd.Index(['A', 'B', 'C']))  
    var_1 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_1 = csr_matrix(np.matrix([[0., 1.], [2., 3.], [4., 5.]]))
   
    obs_2 = pd.DataFrame({'Obs': ['S-C', 'S-A', 'S-B']}, index=pd.Index(['C', 'A', 'B']))  
    var_2 = pd.DataFrame({'Var': ['G-1', 'G-2']}, index=pd.Index(['1', '2']))
    mat_2 = csr_matrix(np.matrix([[4., 5.], [0., 1.], [2., 3.]]))
    
    adata_1 = anndata.AnnData(X=mat_1, obs=obs_1, var=var_1)
    ordered_adata = anndata.AnnData(X=mat_2, obs=obs_2, var=var_2)
    result_adata = AnnDataUtils.reorder_adata(adata_1, pd.Index(['C', 'A', 'B']))
    
    self.assertTrue(
      np.array_equal(
          ordered_adata.X.toarray(), result_adata.X.toarray()), 
          f"The matrix is not sorted correctly.")
    for annot in ['var', 'obs']:
      self.assertTrue(
        getattr(ordered_adata, annot).equals(
          getattr(result_adata, annot)),
        f"The {annot} dataframe is not sorted correctly.")