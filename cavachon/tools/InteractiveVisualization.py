from cavachon.tools.ClusterAnalysis import ClusterAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Union, Sequence

import anndata
import muon as mu
import pandas as pd
import plotly
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
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive barplot.

    Parameters
    ----------
    data: pd.DataFrame
        input data.
    
    x: str
        column names in the data used as variable in X-axis.
    
    y: str
        column names in the data used as variable in Y-axis.

    group: Optional[str], optional
        column names in the data used to color code the groups. 
        Defaults to None.
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
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
      fig.update_layout(barmode='group', **kwargs)
    else:
      means = data.groupby(x).mean()[y]
      sem = data.groupby(x).sem()[y]
      fig = go.Figure()
      fig.add_trace(go.Bar(
          name='Control',
          x=means.index, y=means,
          error_y=dict(type='data', array=sem)))
      fig.update_layout(**kwargs)
    
    return fig

  @staticmethod
  def scatter(*args, **kwargs):
    """Create interactive scatter plot.
   
    Parameters
    ----------
    *args: Optional[Sequence[Any]], optional
        additional positional arguments for px.scatter.

    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for px.scatter.


    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.
    """
    fig = px.scatter(*args, **kwargs)
    fig.update_traces(
        marker=dict(
            opacity=0.7, 
            line=dict(width=1, color='DarkSlateGrey')))
    return fig

  @staticmethod
  def embedding(
      adata: anndata.AnnData,
      method: str = 'tsne',
      filename: Optional[str] = None,
      use_rep: str = 'z',
      color: Union[str, Sequence[str], None] = None,
      title: Optional[str] = None,
      color_discrete_sequence: Optional[Sequence[str]] = None,
      *args,
      **kwargs):
    """Create interactive visualization for the latent space.

    Parameters
    ----------
    adata: anndata.AnnData
        the AnnData used for the analysis.
    
    method: str, optional
        embedding method for the latent space, support 'pca', 'umap' 
        and 'tsne'. Defaults to 'tsne'.
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.
    
    use_rep: str, optional
        which representation to used for the latent space. Defaults to 
        'z'.
    
    color: Union[str, Sequence[str], None], optional
        column names in the adata.obs that used to color code the 
        samples. Alternatively, if provided with `obsm_key`/`obsm_index`
        the color will be set to the `obsm_index` column from the array
        of adata.obsm[`obsm_key`]. The same color for all samples will
        be used if provided with None. Defaults to None.
    
    title: Optional[str], optional
        title for the figure. Defaults to 'Z(name of AnnData)'
    
    color_discrete_sequence: Optional[Sequence[str]], optional
        the discrete color set individually for each sample. This will
        overwrite the color code from `color`. The color code defined
        from `color` argument will be used if provided with none. 
        Defaults to None
    
    *args: Optional[Sequence[Any]], optional
        additional positional arguments for px.scatter.

    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for px.scatter.
    """
    
    adata_name = adata.uns.get('cavachon', '').get('name', '')
    if title is None:
      title = f'Z({adata_name})'

    if method not in ['pca', 'tsne', 'umap']:
      message = ''.join((
          f"Invalid value for method ({method}). Expected to be one of the following: ",
          f"'pca', 'tsne' or 'umap'. Set to 'tsne'."
      ))
      warnings.warn(message, RuntimeWarning)
      method = 'tsne'
     
    if color is not None:

      if color in adata.obs:
        color = adata.obs[color]
      else:
        color_obsm_key = '/'.join(color.split('/')[0:-1])
        color_obsm_column = int(color.split('/')[-1])
        if color_obsm_key in adata.obsm:
          color = adata.obsm[color_obsm_key][:, color_obsm_column]
        else:
          message = ''.join((
            f"{color} is not in adata.obs, and {color_obsm_key} is not in adata.obsm "
            f"ignore color argument."
          ))
          warnings.warn(message, RuntimeWarning)
          color = None

    if color is None:
      if color_discrete_sequence is None:
        color_discrete_sequence = ['salmon'] * adata.n_obs

    obsm_key = f'{use_rep}_{method}'
    if obsm_key not in adata.obsm.keys():
      if method == 'pca':
        model = PCA(n_components=2, random_state=0)
        adata.obsm[obsm_key] = model.fit_transform(adata.obsm[use_rep])
      if method == 'tsne':
        model = TSNE(random_state=0)
        adata.obsm[obsm_key] = model.fit_transform(adata.obsm[use_rep])
      if method == 'umap':
        model = umap.UMAP(random_state=0)
        adata.obsm[obsm_key] = model.fit_transform(adata.obsm[use_rep])
    
    if method == 'pca':
      labels = {'x': 'PCA 1', 'y': 'PCA 2'}
    if method == 'tsne':
      labels = {'x': 't-SNE 1', 'y': 't-SNE 2'}
    if method == 'umap':
      labels = {'x': 'Umap 1', 'y': 'Umap 2'}

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
      **kwargs):
    """Create interactive visualization for nearest neighbor analysis.

    Parameters
    ----------
    mdata: mu.MuData
        the MuData used for the generative process.
    
    model: tf.keras.Model
        the trained generative model used for the generative process.
    
    modality: str
        the modality to used.
    
    use_cluster: str
        the column name of the clusters in the obs of modality.
    
    use_rep: str
        the key of obsm of modality to used to compute the distance 
        within and between clusters
    
    n_neighbors_sequence: Sequence[int], optional
        the number of neighbors to be analyzed, Defaults to 
        list(range(5, 25))
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.
    
    group_by_cluster: bool, optional
        whether or not to group by the clusters. Defaults to False
    
    title: str, optional
        title for the figure. Defaults to 'Cluster Nearest Neighbor
        Analysis'
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.
    """
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
        **kwargs)
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)