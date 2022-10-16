from cavachon.tools.ClusterAnalysis import ClusterAnalysis
from sklearn.decomposition import PCA
from typing import Optional, Union, Sequence

import anndata
import muon as mu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scanpy
import tensorflow as tf
import umap
import warnings

from cavachon.tools.ClusterAnalysis import ClusterAnalysis

class InteractiveVisualization:
  
  @staticmethod
  def bar(
      data: pd.DataFrame,
      x: str,
      y: str,
      group: Optional[str] = None,
      *args, **kwargs):

    if group:
      unique_groups = data[group].value_counts().sort_index().index
      fig = go.Figure()
      for subset in unique_groups:
        data_subset = data.loc[data[group] == subset]
        means = data_subset.groupby(x).mean()[y]
        sem = data_subset.groupby(x).sem()[y]
        fig.add_trace(go.Bar(
            name=subset,
            x=means.index, y=means,
            error_y=dict(type='data', array=sem)))
      fig.update_layout(barmode='group', *args, **kwargs)
    else:
      means = data.groupby(x).mean()[y]
      sem = data.groupby(x).sem()[y]
      fig = go.Figure()
      fig.add_trace(go.Bar(
          name='Control',
          x=means.index, y=means,
          error_y=dict(type='data', array=sem)))
      fig.update_layout(*args, **kwargs)
    
    return fig

  @staticmethod
  def scatter(*args, **kwargs):
    fig = px.scatter(*args, **kwargs)
    fig.update_traces(
        marker=dict(
            opacity=0.7, 
            line=dict(width=1, color='DarkSlateGrey')))
    return fig

  @staticmethod
  def latent_space(
      adata: anndata.AnnData,
      embedding: str = 'tsne',
      filename: Optional[str] = None,
      use_rep: str = 'z',
      color: Union[str, Sequence[str], None] = None,
      title: Optional[str] = None,
      color_discrete_sequence: Optional[Sequence[str]] = None,
      *args,
      **kwargs):
    
    adata_name = adata.uns.get('cavachon', '').get('name', '')
    if title is None:
      title = f'Z({adata_name})'

    if embedding not in ['pca', 'tsne', 'umap']:
      message = ''.join((
          f"Invalid value for embedding ({embedding}). Expected to be one of the following: ",
          f"'pca', 'tsne' or 'umap'. Set to 'tsne'."
      ))
      warnings.warn(message, RuntimeWarning)
      embedding = 'tsne'
    
    obsm_key = f'X_{embedding}'
    if obsm_key not in adata.obsm.keys():
      if embedding == 'pca':
        model = PCA(n_components=2, random_state=0)
        model.fit(adata.obsm[use_rep])
        adata.obsm['X_pca'] = model.transform(adata.obsm[use_rep])
      if embedding == 'tsne':
        scanpy.tl.tsne(adata, use_rep=use_rep)
      if embedding == 'umap':
        model = umap.UMAP(random_state=0)
        model.fit(adata.obsm[use_rep])
        adata.obsm['X_umap'] = model.transform(adata.obsm[use_rep])
    
    if embedding == 'pca':
      labels = {'x': 'PCA 1', 'y': 'PCA 2'}
    if embedding == 'tsne':
      labels = {'x': 't-SNE 1', 'y': 't-SNE 2'}
    if embedding == 'umap':
      labels = {'x': 'Umap 1', 'y': 'Umap 2'}
    
    if color is None:
      if color_discrete_sequence is None:
        color_discrete_sequence = ['salmon'] * adata.n_obs
    else:
        color = adata.obs[color]
        #color_discrete_sequence = None

    x = adata.obsm[obsm_key][:, 0]
    y = adata.obsm[obsm_key][:, 1]
    fig = InteractiveVisualization.scatter(
        x=x,
        y=y,
        labels=labels,
        title=title,
        color=color,
        color_discrete_sequence=color_discrete_sequence,
        *args, **kwargs)
    fig.show()
    
    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)

  @staticmethod
  def neighbors_with_same_annotations(
      mdata: mu.MuData,
      model: tf.keras.Model,
      modality: str,
      use_cluster: str,
      use_rep: str,
      n_neighbors_sequence: Sequence[int] = list(range(5, 25)),
      filename: Optional[str] = None,
      group_by_cluster: bool = False,
      title: str = 'Cluster Nearest Neighbor Analysis',
      *args,
      **kwargs):
    analysis = ClusterAnalysis(mdata, model)
    analysis_result = analysis.compute_neighbors_with_same_annotations(
        modality=modality, 
        use_cluster=use_cluster,
        use_rep=use_rep,
        n_neighbors_sequence=n_neighbors_sequence)
    
    if group_by_cluster:
      group = 'Cluster'
    else:
      group = None

    fig = InteractiveVisualization.bar(
        analysis_result, 
        x='Number of Neighbors', 
        y='% of KNN Cells with the Same Cluster',
        group=group,
        title=title,
        xaxis_title='Number of Neighbors',
        yaxis_title='% of KNN Cells with the Same Cluster',
        *args,
        **kwargs)
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)