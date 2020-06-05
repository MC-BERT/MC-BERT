# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
MC-BERT: Efficient Language Pre-Training via a Meta Controller
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .hub_interface import McbertHubInterface

@register_model('mcbert')
class McbertModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):
        return {
        }

    def __init__(self, args, mc_encoder, gen_encoder, neither_idx):
        super().__init__(gen_encoder)
        self.mc = mc_encoder
        self.args = args
        self.neither_idx = neither_idx

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--mc-size-divider', type=int,
                            help='divider for mc: layer size, FFN size and attention heads')
        parser.add_argument('--class-num', type=int, default=10,
                            help='total number of classes')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--embedding-normalize', action='store_true',
                            help='add layernorm after the embedding layer')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        parser.add_argument('--debug', action='store_true',
                            help='trigger for pdb')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        args.vocab_num = len(task.source_dictionary)
        args.vocab_nspecial = task.source_dictionary.nspecial

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        gen_encoder = GenEncoder(args, task.source_dictionary)
        if args.task == 'mcbert':
            mc_encoder = MCEncoder(args, task.source_dictionary, gen_encoder.sentence_encoder.embed_tokens, gen_encoder.lm_head.bias.weight)
        else:
            mc_encoder = None
        return cls(args, mc_encoder, gen_encoder, task.source_dictionary.index('<neither>'))

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, classification_head_name=None, masked_tokens=None, targets=None, **kwargs):
        if classification_head_name is not None:
            features_only = True

        if self.args.task == 'mcbert':
            mc_x_mask, features, _ = self.mc(
                src_tokens,
                features_only=False,
                return_all_hiddens=False,
                masked_tokens=masked_tokens,
                return_top_features=True,
                **kwargs
            )  # Float[num_masked, vocab]

            with torch.no_grad():
                mc_x_all = self.mc.lm_head(features, masked_tokens=None)
                sample_probs = torch.softmax(mc_x_all.detach(), -1, dtype=torch.float32)  # Float[bs, seq_len, vocab]
                sample_probs = sample_probs.view(-1, sample_probs.size(-1))  # Float[bs * seq_len, vocab]
                sampled_tokens = torch.multinomial(
                    sample_probs, self.args.class_num, replacement=True
                ).view(src_tokens.size(0), src_tokens.size(1), self.args.class_num)  # Float[bs, seq_len, class_num]

                src_tokens = src_tokens.clone()
                if masked_tokens is not None:  # masked_tokens : Float[bs, seq_len]
                    src_tokens[masked_tokens] = sampled_tokens[:, :, 0][masked_tokens]

                # for replaced tokens, the correct label is 1-th element (the target)
                replace_tokens = src_tokens.ne(targets)
                sampled_tokens[:, :, 1][replace_tokens] = targets[replace_tokens]

                # for non-replaced tokens, the correct label is 0-th element (the <neither> token)
                sampled_tokens[:, :, 0] = self.neither_idx

                # mcerate targets according to the rule mentioned above
                gen_target = replace_tokens.long()

        gen_x, extra = self.decoder(
            src_tokens,
            features_only=features_only,
            return_all_hiddens=return_all_hiddens,
            masked_tokens=None,  # for generator, predict all of the tokens
            out_helper=sampled_tokens if self.args.task == 'mcbert' else None,
            **kwargs
        )

        if classification_head_name is not None:
            gen_x = self.classification_heads[classification_head_name](gen_x)
        
        if self.args.task == 'mcbert':
            return mc_x_mask, gen_x, src_tokens, gen_target, extra
        else:
            return gen_x, extra

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = McbertClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @property
    def supported_targets(self):
        return {'self'}

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return McbertHubInterface(x['args'], x['task'], x['models'][0])

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v


class MCLMHead(nn.Module):
    """Head for Meta Controller"""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None, share_emb_pro=None, bias=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        elif share_emb_pro is not None:
            self.add_one_linear = True
            self.share_emb_pro = share_emb_pro
        else:
            self.add_one_linear = False
        self.weight = weight

        if bias is None:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        self.bias = bias

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]
 
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        if self.add_one_linear:
            weight = self.share_emb_pro(self.weight)
        else:
            weight = self.weight
        # project back to size of vocabulary with bias
        x = F.linear(x, weight) + self.bias.view(-1)
        return x


class McbertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MCEncoder(FairseqDecoder):
    """MC encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary, share_embed_tokens, lmhead_bias_helper):
        super().__init__(dictionary)
        self.args = args
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=int(args.encoder_embed_dim / args.mc_size_divider),
            ffn_embedding_dim=int(args.encoder_ffn_embed_dim / args.mc_size_divider),
            num_attention_heads=int(args.encoder_attention_heads / args.mc_size_divider),
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=args.encoder_normalize_before,
            embedding_normalize=args.embedding_normalize,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            share_embed_tokens=share_embed_tokens,
            shared_embedding_dim=args.encoder_embed_dim,
        )
        self.lm_head = MCLMHead(
            embed_dim=int(args.encoder_embed_dim / args.mc_size_divider),
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.embed_tokens.weight,
            share_emb_pro=self.sentence_encoder.embed_linear,
            bias=lmhead_bias_helper,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, return_top_features=False, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens)
        if return_top_features:
            features = x
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        if return_top_features:
            return x, features, extra
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1]
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


class GenEncoder(FairseqDecoder):
    """McbertModel generator encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)
        self.args = args
        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=args.encoder_normalize_before,
            embedding_normalize=args.embedding_normalize,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
        )
        self.lm_head = GenLMHead(
            embed_dim=int(args.encoder_embed_dim),
            output_dim=args.class_num,
            activation_fn=args.activation_fn,
            embed_tokens=self.sentence_encoder.embed_tokens,
        )

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, masked_tokens=None, out_helper=None, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """
        x, extra = self.extract_features(src_tokens, return_all_hiddens)
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens, out_helper=out_helper)
        return x, extra

    def extract_features(self, src_tokens, return_all_hiddens=False, **unused):
        inner_states, _ = self.sentence_encoder(
            src_tokens,
            last_state_only=not return_all_hiddens,
        )
        features = inner_states[-1]
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, out_helper=None, **unused):
        return self.lm_head(features, masked_tokens, out_helper)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

class GenLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, embed_tokens):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        self.embed_tokens = embed_tokens
        self.bias = nn.Embedding(embed_tokens.num_embeddings, 1)

    def forward(self, features, masked_tokens=None, out_helper=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]
 
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)

        if out_helper is not None:
            # x: Float[bs, seq_len, dim]
            weight = self.embed_tokens(out_helper).transpose(2, 3)  # Float[bs, seq_len, dim, num_class]
            bias = self.bias(out_helper).transpose(2, 3)  # Float[bs, seq_len, 1, num_class]

            bs = x.size(0)
            x = torch.baddbmm(
                input=bias.view(-1, bias.size(-2), bias.size(-1)),
                batch1=x.view(-1, 1, x.size(-1)),
                batch2=weight.view(-1, weight.size(-2), weight.size(-1))
            )  # Float[bs * seq_len, 1, num_class]

            return x.view(bs, -1, x.size(-1))  # Float[bs, seq_len, num_class]
        else:
            return x


@register_model_architecture('mcbert', 'mcbert')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.mc_size_divider = getattr(args, 'mc_size_divider', 3)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)

    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.embedding_normalize = getattr(args, 'embedding_normalize', False)


@register_model_architecture('mcbert', 'mcbert_base')
def mcbert_base_architecture(args):
    base_architecture(args)


@register_model_architecture('mcbert', 'mcbert_small')
def mcbert_small_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 12)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.mc_size_divider = getattr(args, 'mc_size_divider', 2)

    base_architecture(args)


@register_model_architecture('mcbert', 'mcbert_large')
def mcbert_small_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.mc_size_divider = getattr(args, 'mc_size_divider', 4)

    base_architecture(args)
