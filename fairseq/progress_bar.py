# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wrapper around various loggers and progress bars (e.g., tqdm).
"""

from collections import OrderedDict
import json
from numbers import Number
import os
import sys

from fairseq import distributed_utils
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter

g_tbmf_wrapper = None


def build_progress_bar(args, iterator, epoch=None, prefix=None, default='tqdm', no_progress_bar='none'):
    if args.log_format is None:
        args.log_format = no_progress_bar if args.no_progress_bar else default

    if args.log_format == 'tqdm' and not sys.stderr.isatty():
        args.log_format = 'simple'

    if args.log_format == 'json':
        bar = json_progress_bar(iterator, epoch, prefix, args.log_interval)
    elif args.log_format == 'none':
        bar = noop_progress_bar(iterator, epoch, prefix)
    elif args.log_format == 'simple':
        bar = simple_progress_bar(iterator, epoch, prefix, args.log_interval)
    elif args.log_format == 'tqdm':
        bar = tqdm_progress_bar(iterator, epoch, prefix)
    else:
        raise ValueError('Unknown log format: {}'.format(args.log_format))

    if args.tbmf_wrapper and distributed_utils.is_master(args):
        global g_tbmf_wrapper
        if g_tbmf_wrapper is None:
            try:
                from fairseq.fb_tbmf_wrapper import fb_tbmf_wrapper
            except Exception:
                raise ImportError("fb_tbmf_wrapper package not found.")
            g_tbmf_wrapper = fb_tbmf_wrapper
        bar = g_tbmf_wrapper(bar, args, args.log_interval)
    elif args.tensorboard_logdir and distributed_utils.is_master(args):
        bar = tensorboard_log_wrapper(bar, args.tensorboard_logdir, args)

    return bar


def format_stat(stat):
    if isinstance(stat, Number):
        stat = '{:g}'.format(stat)
    elif isinstance(stat, AverageMeter):
        stat = '{:.3f}'.format(stat.avg)
    elif isinstance(stat, TimeMeter):
        stat = '{:g}'.format(round(stat.avg))
    elif isinstance(stat, StopwatchMeter):
        stat = '{:g}'.format(round(stat.sum))
    return stat


class progress_bar(object):
    """Abstract class for progress bars."""
    def __init__(self, iterable, epoch=None, prefix=None):
        self.iterable = iterable
        self.offset = getattr(iterable, 'offset', 0)
        self.epoch = epoch
        self.prefix = ''
        if epoch is not None:
            self.prefix += '| epoch {:03d}'.format(epoch)
        if prefix is not None:
            self.prefix += ' | {}'.format(prefix)

    def __len__(self):
        return len(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        raise NotImplementedError

    def log(self, stats, tag='', step=None):
        """Log intermediate stats according to log_interval."""
        raise NotImplementedError

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        raise NotImplementedError

    def _str_commas(self, stats):
        return ', '.join(key + '=' + stats[key].strip()
                         for key in stats.keys())

    def _str_pipes(self, stats):
        return ' | '.join(key + ' ' + stats[key].strip()
                          for key in stats.keys())

    def _format_stats(self, stats):
        postfix = OrderedDict(stats)
        # Preprocess stats according to datatype
        for key in postfix.keys():
            postfix[key] = str(format_stat(postfix[key]))
        return postfix


class json_progress_bar(progress_bar):
    """Log output in JSON format."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.stats = None

    def __iter__(self):
        size = float(len(self.iterable))
        for i, obj in enumerate(self.iterable, start=self.offset):
            yield obj
            if self.stats is not None and i > 0 and \
                    self.log_interval is not None and i % self.log_interval == 0:
                update = self.epoch - 1 + float(i / size) if self.epoch is not None else None
                stats = self._format_stats(self.stats, epoch=self.epoch, update=update)
                print(json.dumps(stats), flush=True)

    def log(self, stats, tag='', step=None):
        """Log intermediate stats according to log_interval."""
        self.stats = stats

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        self.stats = stats
        if tag != '':
            self.stats = OrderedDict([(tag + '_' + k, v) for k, v in self.stats.items()])
        stats = self._format_stats(self.stats, epoch=self.epoch)
        print(json.dumps(stats), flush=True)

    def _format_stats(self, stats, epoch=None, update=None):
        postfix = OrderedDict()
        if epoch is not None:
            postfix['epoch'] = epoch
        if update is not None:
            postfix['update'] = round(update, 3)
        # Preprocess stats according to datatype
        for key in stats.keys():
            postfix[key] = format_stat(stats[key])
        return postfix


class noop_progress_bar(progress_bar):
    """No logging."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)

    def __iter__(self):
        for obj in self.iterable:
            yield obj

    def log(self, stats, tag='', step=None):
        """Log intermediate stats according to log_interval."""
        pass

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        pass


class simple_progress_bar(progress_bar):
    """A minimal logger for non-TTY environments."""

    def __init__(self, iterable, epoch=None, prefix=None, log_interval=1000):
        super().__init__(iterable, epoch, prefix)
        self.log_interval = log_interval
        self.stats = None

    def __iter__(self):
        size = len(self.iterable)
        for i, obj in enumerate(self.iterable, start=self.offset):
            yield obj
            if self.stats is not None and i > 0 and \
                    self.log_interval is not None and i % self.log_interval == 0:
                postfix = self._str_commas(self.stats)
                print('{}:  {:5d} / {:d} {}'.format(self.prefix, i, size, postfix),
                      flush=True)

    def log(self, stats, tag='', step=None):
        """Log intermediate stats according to log_interval."""
        self.stats = self._format_stats(stats)

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        print('{} | {}'.format(self.prefix, postfix), flush=True)


class tqdm_progress_bar(progress_bar):
    """Log to tqdm."""

    def __init__(self, iterable, epoch=None, prefix=None):
        super().__init__(iterable, epoch, prefix)
        from tqdm import tqdm
        self.tqdm = tqdm(iterable, self.prefix, leave=False)

    def __iter__(self):
        return iter(self.tqdm)

    def log(self, stats, tag='', step=None):
        """Log intermediate stats according to log_interval."""
        self.tqdm.set_postfix(self._format_stats(stats), refresh=False)

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        postfix = self._str_pipes(self._format_stats(stats))
        self.tqdm.write('{} | {}'.format(self.tqdm.desc, postfix))


class tensorboard_log_wrapper(progress_bar):
    """Log to tensorboard."""

    def __init__(self, wrapped_bar, tensorboard_logdir, args):
        self.wrapped_bar = wrapped_bar
        self.tensorboard_logdir = tensorboard_logdir
        self.args = args

        try:
            from tensorboardX import SummaryWriter
            self.SummaryWriter = SummaryWriter
            self._writers = {}
        except ImportError:
            print("tensorboard or required dependencies not found, "
                  "please see README for using tensorboard. (e.g. pip install tensorboardX)")
            self.SummaryWriter = None

    def _writer(self, key):
        if self.SummaryWriter is None:
            return None
        if key not in self._writers:
            self._writers[key] = self.SummaryWriter(
                os.path.join(self.tensorboard_logdir, key),
            )
            self._writers[key].add_text('args', str(vars(self.args)))
            self._writers[key].add_text('sys.argv', " ".join(sys.argv))
        return self._writers[key]

    def __iter__(self):
        return iter(self.wrapped_bar)

    def log(self, stats, tag='', step=None):
        """Log intermediate stats to tensorboard."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.log(stats, tag=tag, step=step)

    def print(self, stats, tag='', step=None):
        """Print end-of-epoch stats."""
        self._log_to_tensorboard(stats, tag, step)
        self.wrapped_bar.print(stats, tag=tag, step=step)

    def __exit__(self, *exc):
        for writer in getattr(self, '_writers', {}).values():
            writer.close()
        return False

    def _log_to_tensorboard(self, stats, tag='', step=None):
        writer = self._writer(tag)
        if writer is None:
            return
        if step is None:
            step = stats['num_updates']
        for key in stats.keys() - {'num_updates'}:
            if isinstance(stats[key], AverageMeter):
                writer.add_scalar(key, stats[key].val, step)
            elif isinstance(stats[key], Number):
                writer.add_scalar(key, stats[key], step)
