from sklearn.decomposition import PCA
from typing import Optional, Union, Sequence

import anndata
import plotly.express as px
import scanpy
import umap
import warnings

class InteractiveVisualization:
  
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
      color: Union[str, Sequence[str], None] = None,
      title: Optional[str] = None,
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
        model.fit(adata.obsm['z'])
        adata.obsm['X_pca'] = model.transform(adata.obsm['z'])
      if embedding == 'tsne':
        scanpy.tl.tsne(adata, use_rep='z')
      if embedding == 'umap':
        model = umap.UMAP(random_state=0)
        model.fit(adata.obsm['z'])
        adata.obsm['X_umap'] = model.transform(adata.obsm['z'])
    
    if embedding == 'pca':
      labels = {'x': 'PCA 1', 'y': 'PCA 2'}
    if embedding == 'tsne':
      labels = {'x': 't-SNE 1', 'y': 't-SNE 2'}
    if embedding == 'umap':
      labels = {'x': 'Umap 1', 'y': 'Umap 2'}
    
    if color is None:
      color_discrete_sequence = ['salmon'] * adata.n_obs
    else:
      color = adata.obs[color]
      color_discrete_sequence = None

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
