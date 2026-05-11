# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for variant scoring."""

import anndata
import numpy as np
import pandas as pd


def _validate_track_metadata(metadata: pd.DataFrame) -> None:
  """Validate the track metadata positive and negative track order."""
  tracks_by_strand = metadata.groupby('strand')

  def _get_group(name: str) -> pd.DataFrame:
    return (
        tracks_by_strand.get_group(name).reset_index()
        if name in tracks_by_strand.groups
        else pd.DataFrame(columns=['name'])
    )

  positive_track_names = _get_group('+')['name']
  negative_track_names = _get_group('-')['name']
  if not positive_track_names.equals(negative_track_names):
    raise ValueError(
        'Positive and negative tracks do not match.'
        f' positive_track_names: {positive_track_names.to_list()},'
        f' negative_track_names: {negative_track_names.to_list()}.'
    )


def _calculate_track_masks(
    track_metadata: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
  """Calculates the positive and negative track masks for the given metadata."""

  # Include unstranded tracks in both positive and negative masks for merging.
  positive_track_mask = (track_metadata['strand'] != '-').values
  negative_track_mask = (track_metadata['strand'] != '+').values

  stranded_track_names = set(
      track_metadata[track_metadata['strand'] != '.']['name']
  )
  duplicate_unstranded_track_mask = (
      (track_metadata['strand'] == '.')
      & track_metadata['name'].isin(stranded_track_names)
  ).values
  positive_track_mask &= ~duplicate_unstranded_track_mask
  negative_track_mask &= ~duplicate_unstranded_track_mask
  return positive_track_mask, negative_track_mask


def merge_stranded_track_metadata(
    track_metadata: pd.DataFrame,
    *,
    validate_tracks: bool = True,
) -> pd.DataFrame:
  """Merges stranded track metadata into a single unstranded track.

  Any existing unstranded tracks with the same name as the merged stranded track
  will be dropped to avoid duplication.

  Args:
    track_metadata: The DataFrame containing the track metadata.
    validate_tracks: If True, validate that the positive and negative tracks
      have the same ordered names.

  Returns:
    The DataFrame with the merged stranded track metadata.
  """

  if not {'name', 'strand'}.issubset(track_metadata.columns):
    raise ValueError('Track metadata must contain "name" and "strand" columns.')

  if validate_tracks:
    _validate_track_metadata(track_metadata)

  track_mask, _ = _calculate_track_masks(track_metadata)

  merged_metadata = track_metadata[track_mask].reset_index(drop=True)
  merged_metadata['strand'] = '.'
  return merged_metadata


def merge_stranded_gene_tracks(
    scores: anndata.AnnData, *, validate_tracks: bool = True
) -> anndata.AnnData:
  """Merges gene scores from stranded tracks into a single unstranded track.

  Gene and splice site variant scorers are strand specific, so scores for
  strand-mismatched genes and tracks are NaN. This function merges the scores
  for separate positive and negative tracks into a single, unstranded track. Any
  existing unstranded tracks with the same name as the merged stranded track
  will be dropped to avoid duplication.

  This function assumes that positive and negative tracks have the same
  ordering.

  If the scores do not have a strand column in the observation metadata, this
  function will return the scores unchanged.

  Args:
    scores: The AnnData containing the gene scores and metadata.
    validate_tracks: If True, validate that the positive and negative tracks
      have the same ordered names.

  Returns:
    An AnnData object with the merged gene scores and metadata.
  """

  if not {'gene_id', 'strand'}.issubset(scores.obs.columns):
    return scores

  if not {'name', 'strand'}.issubset(scores.var.columns):
    raise ValueError('Track metadata must contain "name" and "strand" columns.')

  if validate_tracks:
    _validate_track_metadata(scores.var)

  positive_track_mask, negative_track_mask = _calculate_track_masks(scores.var)

  gene_strands = (scores.obs['strand'] == '+').values

  def _merge_scores(values: np.ndarray) -> np.ndarray:
    positive_track_scores = values[:, positive_track_mask]
    negative_track_scores = values[:, negative_track_mask]

    return np.where(
        gene_strands[:, np.newaxis],
        positive_track_scores,
        negative_track_scores,
    )

  merged_scores = _merge_scores(scores.X)
  merged_layers = {}
  for k, v in scores.layers.items():
    merged_layers[k] = _merge_scores(v)

  merged_metadata = scores.var[positive_track_mask].reset_index(drop=True)
  merged_metadata['strand'] = '.'
  merged_metadata.index = merged_metadata.index.map(str)

  return anndata.AnnData(
      merged_scores,
      obs=scores.obs,
      var=merged_metadata,
      layers=merged_layers,
  )
