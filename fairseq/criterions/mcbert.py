# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('mcbert')
class McbertLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        mask_idx = self.task.dictionary.index('<mask>')
        masked_tokens = sample['net_input']['src_tokens'].eq(mask_idx)
        
        not_pad_tokens = sample['target'].ne(self.padding_idx)
        
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        gen_logits, disc_output, disc_tokens, disc_target, _ = model(
            **sample['net_input'],
            masked_tokens=masked_tokens,
            targets=sample['target']
        )
        disc_output = disc_output[not_pad_tokens]
        disc_target = disc_target[not_pad_tokens]
        targets = model.get_targets(sample, [gen_logits])

        if sample_size != 0:
            targets = targets[masked_tokens]

        gen_loss = F.nll_loss(
            F.log_softmax(
                gen_logits.view(-1, gen_logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        disc_loss = F.nll_loss(
            F.log_softmax(
                disc_output.view(-1, disc_output.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            disc_target.view(-1),
            reduction='sum',
        )

        disc_sample_size = not_pad_tokens.int().sum().item()

        loss = gen_loss + self.args.loss_lamda * disc_loss * sample_size / disc_sample_size

        # import pdb
        # pdb.set_trace()
        with torch.no_grad():
            tp = ((torch.argmax(disc_output.float(), -1).view(-1) == disc_target) & (disc_target == 0)).long().sum()
            fp = ((torch.argmax(disc_output.float(), -1).view(-1) != disc_target) & (disc_target == 1)).long().sum()
            fn = ((torch.argmax(disc_output.float(), -1).view(-1) != disc_target) & (disc_target == 0)).long().sum()
            tn = ((torch.argmax(disc_output.float(), -1).view(-1) == disc_target) & (disc_target == 1)).long().sum()
            assert (tp + fp + tn + fn) == disc_target.size(0), 'invalid size'

        logging_output = {
            'loss': utils.item(disc_loss.data) if reduce else disc_loss.data,
            'nll_loss': utils.item(gen_loss.data) if reduce else gen_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'disc_sample_size': disc_sample_size,
        }
        logging_output.update(
            disc_loss=disc_loss.item()
        )
        logging_output.update(
            gen_loss=gen_loss.item()
        )
        logging_output.update(tp = utils.item(tp.data) if reduce else tp.data)
        logging_output.update(fp = utils.item(fp.data) if reduce else fp.data)
        logging_output.update(fn = utils.item(fn.data) if reduce else fn.data)
        logging_output.update(tn = utils.item(tn.data) if reduce else tn.data)
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        disc_sample_size = sum(log.get('disc_sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / disc_sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'disc_sample_size': disc_sample_size,
        }

        if 'tp' in logging_outputs[0]: 
            tp_sum = sum(log.get('tp', 0) for log in logging_outputs)
            fp_sum = sum(log.get('fp', 0) for log in logging_outputs)
            fn_sum = sum(log.get('fn', 0) for log in logging_outputs)
            tn_sum = sum(log.get('tn', 0) for log in logging_outputs)
            assert tp_sum + fp_sum + fn_sum + tn_sum == disc_sample_size, 'invalid size when aggregating'
            bin_acc = (tp_sum + tn_sum) / disc_sample_size
            replace_acc = tn_sum / (tn_sum + fp_sum + 1e-5)
            non_replace_acc = tp_sum / (tp_sum + fn_sum + 1e-5)
            agg_output.update(bin_acc=bin_acc)
            agg_output.update(replace_acc=replace_acc)
            agg_output.update(non_replace_acc=non_replace_acc)
            agg_output.update(replace_samples=(tn_sum + fp_sum))
            agg_output.update(replace_rate=(tn_sum + fp_sum)/disc_sample_size)

        disc_loss = sum(log.get('disc_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(disc_loss=disc_loss)
        gen_loss = sum(log.get('gen_loss', 0) for log in logging_outputs) / len(logging_outputs)
        agg_output.update(gen_loss=gen_loss)
        
        return agg_output
