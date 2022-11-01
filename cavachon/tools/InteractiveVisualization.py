from cavachon.tools.ClusterAnalysis import ClusterAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, Mapping, Union, Sequence

import anndata
import numpy as np
import muon as mu
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
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
      color_discrete_map: Mapping[str, str] = dict(),
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
        if color_discrete_map.get(subset, None):
          fig.add_trace(go.Bar(
              name=subset,
              x=means.index, 
              y=means, 
              marker_opacity=0.7,
              marker_line=dict(width=1, color='DarkSlateGrey'),
              marker_color=color_discrete_map.get(subset),
              error_y=dict(type='data', array=sem)))
        else:
            fig.add_trace(go.Bar(
              name=subset,
              x=means.index, 
              y=means,
              marker_opacity=0.7,
              marker_line=dict(width=1, color='DarkSlateGrey'),
              error_y=dict(type='data', array=sem)))
      fig.update_layout(barmode='group', **kwargs)
    else:
      means = data.groupby(x).mean()[y]
      sem = data.groupby(x).sem()[y]
      fig = go.Figure()
      fig.add_trace(go.Bar(
          name='Control',
          x=means.index, y=means,
          marker_opacity=0.7,
          marker_line=dict(width=1, color='DarkSlateGrey'),
          error_y=dict(type='data', array=sem)))

    return fig

  @staticmethod
  def scatter(*args, **kwargs) -> plotly.graph_objs._figure.Figure:
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
      use_rep: Union[str, np.array] = 'z',
      color: Union[str, Sequence[str], None] = None,
      title: Optional[str] = None,
      color_discrete_sequence: Optional[Sequence[str]] = None,
      force: bool = False,
      *args,
      **kwargs) -> plotly.graph_objs._figure.Figure:
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
    
    use_rep: Union[str, np.array], optional
        which representation to used for the latent space. Defaults to 
        'z'. Alternatively, the array will be used if provided with 
        np.array,
    
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
        from `color` argument will be used if provided with none. To 
        change the color palette for `color`, use `color_discrete_map`
        instead. Defaults to None.
    
    force: bool, optional
        force to rerun the embedding. Defaults to False.
    
    *args: Optional[Sequence[Any]], optional
        additional positional arguments for px.scatter.

    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for px.scatter.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

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
    
    if isinstance(use_rep, np.ndarray):
      matrix = use_rep
    else:
      matrix = adata.obsm[use_rep]
    
    # np.ndarray has __str__ implemented. It also works if use_rep is a
    # np.ndarray
    obsm_key = f'{use_rep}_{method}'  
    if force or obsm_key not in adata.obsm.keys():
      if method == 'pca':
        model = PCA(n_components=2, random_state=0)
        transformed_matrix = model.fit_transform(matrix)
      if method == 'tsne':
        model = TSNE(random_state=0)
        transformed_matrix = model.fit_transform(matrix)
      if method == 'umap':
        model = umap.UMAP(random_state=0)
        transformed_matrix = model.fit_transform(matrix)
      
      if isinstance(use_rep, str):
        adata.obsm[obsm_key] = transformed_matrix

    # in case the embedding is not called again.
    if isinstance(use_rep, str):
      transformed_matrix = adata.obsm[obsm_key]

    x = transformed_matrix[:, 0]
    y = transformed_matrix[:, 1]

    if method == 'pca':
      labels = {'x': 'PCA 1', 'y': 'PCA 2'}
    if method == 'tsne':
      labels = {'x': 't-SNE 1', 'y': 't-SNE 2'}
    if method == 'umap':
      labels = {'x': 'Umap 1', 'y': 'Umap 2'}

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
    
    return fig

  @staticmethod
  def neighbors_with_same_annotations(
      mdata: mu.MuData,
      model: tf.keras.Model,
      modality: str,
      use_cluster: str,
      use_rep: Union[str, np.array],
      n_neighbors: Sequence[int] = list(range(5, 25)),
      filename: Optional[str] = None,
      group_by_cluster: bool = False,
      color_discrete_map: Mapping[str, str] = dict(),
      title: str = 'Cluster Nearest Neighbor Analysis',
      **kwargs) -> plotly.graph_objs._figure.Figure:
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
    
    use_rep: Union[str, np.array]
        the key of obsm of modality to used to compute the distance 
        within and between clusters. Alternatively, the array will be 
        used if provided with np.array,
    
    n_neighbors: Union[int, Sequence[int]], optional
        the number of neighbors to be analyzed, Defaults to 
        list(range(5, 25))
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.
    
    group_by_cluster: bool, optional
        whether or not to group by the clusters. Defaults to False
    
    color_discrete_map: Mapping[str, str], optional
        the color palette for `group_by_cluster`. Defaults to dict()

    title: str, optional
        title for the figure. Defaults to 'Cluster Nearest Neighbor
        Analysis'
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
    analysis = ClusterAnalysis(mdata, model)
    analysis_result = analysis.compute_neighbors_with_same_annotations(
        modality=modality, 
        use_cluster=use_cluster,
        use_rep=use_rep,
        n_neighbors=n_neighbors)
    
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
        color_discrete_map=color_discrete_map,
        xaxis_title='Number of Neighbors',
        yaxis_title='% of KNN Cells with the Same Cluster',
        **kwargs)
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig
  
  @staticmethod
  def attribution_score(
      mdata: mu.MuData,
      model: tf.keras.Model,
      component: str,
      modality: str,
      target_component: str,
      use_cluster: str,
      steps: int = 10,
      batch_size: int = 128,
      color_discrete_map: Mapping[str, str] = dict(),
      filename: Optional[str] = None,
      title: str = 'Attribution Score Analysis',
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for attribution score.

    Parameters
    ----------
    mdata: mu.MuData
        the MuData used for the generative process.
    
    model: tf.keras.Model
        the trained generative model used for the generative process.

    component: str
        the outputs of which component to used.

    modality: str
        which modality of the outputs of the component to used.

    target_component: str
        the latent representation of which component to used.

    use_cluster: str
        the column name of the clusters in the obs of modality.

    steps: int, optional
        steps in integrated gradients. Defaults to 10.

    batch_size: int, optional
        batch size used for the forward pass. Defaults to 128.

    color_discrete_map: Mapping[str, str], optional
        the color palette for `group_by_cluster`. Defaults to dict()

    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.

    title: str, optional
        title for the figure. Defaults to 'Attribution Score Analysis'
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """

    analysis = AttributionAnalysis(mdata, model)
    attribution_score = analysis.compute_integrated_gradient(
        component=component,
        modality=modality,
        target_component=target_component,
        steps=steps,
        batch_size=batch_size)

    data = pd.DataFrame({
        'X': np.ones(workflow.mdata[modality].n_obs),
        'Cluster': workflow.mdata[modality].obs[use_cluster], 
        'Attribution Score': np.mean(np.abs(attribution_score), axis=-1)})

    fig = InteractiveVisualization.bar(
        data, 
        x='X', 
        y='Attribution Score',
        group='Cluster',
        title=title,
        color_discrete_map=color_discrete_map,
        xaxis_title='Cluster',
        yaxis_title='Attribution Score',
        **kwargs)
    fig.update_layout(xaxis_showticklabels=False)
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig
  
  @staticmethod
  def differential_volcano_plot(
      mdata: mu.MuData,
      model: tf.keras.Model,
      group_a_index: Union[pd.Index, Sequence[str]],
      group_b_index: Union[pd.Index, Sequence[str]],
      component: str,
      modality: str,
      significant_threshold: float = 3.2,
      filename: Optional[str] = None,
      title: str = 'Volcano Plot for Differential Analysis',
      **kwargs) -> plotly.graph_objs._figure.Figure:
    """Create interactive visualization for volcano plot for 
    differential analysis

    Parameters
    ----------
    mdata: mu.MuData
        the MuData used for the generative process.
    
    model: tf.keras.Model
        the trained generative model used for the generative process.

    group_a_index : Union[pd.Index, Sequence[str]]
        index of group one. Needs to meet the index in the obs of the
        modality.
    
    group_b_index : Union[pd.Index, Sequence[str]]
        index of group two. Needs to meet the index in the obs of the
        modality.
    
    component : str
        generative result of `modality` from which component to used.
    
    modality : str
        which modality to used from the generative result of 
        `component`.

    significant_threshold : float, optional
        threshold for significance of Bayesian factor. Defaults to 3.2.
    
    filename: Optional[str], optional
        filename to save the figure. None if disable saving. Defaults 
        to None.

    title: str, optional
        title for the figure. Defaults to 'Attribution Score Analysis'
    
    **kwargs: Optional[Mapping[str, Any]], optional
        additional arguments for fig.update_layout.
    
    Returns
    -------
    plotly.graph_objs._figure.Figure
        interactive figure objects.

    """
    obs = mdata[modality].obs
    analysis = DifferentialAnalysis(mdata=workflow.mdata, model=workflow.model)
    degs = analysis.between_two_groups(
        group_a_index=group_a_index,
        group_b_index=group_b_index, 
        component=component,
        modality=modality)
        
    degs['LogFC(A/B)'] = np.log(degs['Mean(A)']/degs['Mean(B)'])
    degs['K(A!=B|Z)'] = degs[['K(A>B|Z)', 'K(B>A|Z)']].abs().max(axis=1)
    degs['Significant'] = degs['K(A!=B|Z)'].apply(lambda x: 'Significant' if x > 3.2 else 'Non-significant')
    degs['Target'] = degs.index
    fig = InteractiveVisualization.scatter(
        degs,
        x='LogFC(A/B)',
        y='K(A!=B|Z)',
        color='Significant', 
        labels={'x': 'LogFC(A/B)', 'y': 'K(A!=B|Z)'},
        hover_data=['Target'],
        title=title,
        **kwargs)
    
    fig.show()

    if filename:
      if filename.endswith('html'):
        fig.write_html(filename)
      else:
        fig.write_image(filename)
    
    return fig