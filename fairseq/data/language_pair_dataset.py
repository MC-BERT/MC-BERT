# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import data_utils, FairseqDataset


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            print("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order), dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }
        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
