"""Performance knobs for MiN runs.

Everything here is **opt-in and defaults to current behaviour**, so importing and
calling ``apply_perf_settings`` on an unmodified config is a no-op.  This exists
because a DINOv3-H+/16 iNat21 cell profiles out at ~52 TFLOPS sustained on an
H100 -- roughly 78% of the *FP32 non-tensor-core* peak (67 TFLOPS) and ~10% of
the TF32 tensor-core peak (~495 TFLOPS).  MiN never touches the float32 matmul
precision, so PyTorch's default (``allow_tf32 = False`` since 1.12) leaves the
tensor cores essentially idle for the entire run.

Config keys (all optional):

``matmul_precision``
    One of ``"highest"`` (default, unchanged IEEE fp32), ``"high"`` (TF32),
    ``"medium"`` (bf16 inputs).  Maps onto
    ``torch.set_float32_matmul_precision``.  ``"high"`` keeps fp32 storage and
    fp32 accumulation and only rounds GEMM *inputs* from a 23- to a 10-bit
    mantissa.

``cudnn_benchmark``
    ``true`` to let cuDNN autotune the patch-embedding convolution.  Safe for
    MiN because every batch after the first is a fixed shape.

``dataloader``
    Dict of DataLoader overrides, e.g.
    ``{"num_workers": 12, "persistent_workers": true, "prefetch_factor": 4,
       "pin_memory": true}``.  Consumed by :func:`loader_kwargs`.

None of these change what MiN computes in exact arithmetic, but
``matmul_precision`` is a floating-point change and must clear the validation
gate (reproduce a known-good cell) before it is trusted.
"""

import torch

_VALID_PRECISIONS = ('highest', 'high', 'medium')


def apply_perf_settings(args, logger=None):
    """Apply opt-in performance settings from ``args``.  No-op by default."""

    def _log(msg):
        print(msg)
        if logger is not None:
            logger.info(msg)

    precision = args.get('matmul_precision', 'highest')
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            "matmul_precision must be one of {}, got {!r}".format(
                _VALID_PRECISIONS, precision))
    if precision != 'highest':
        torch.set_float32_matmul_precision(precision)
        _log('[perf] float32 matmul precision -> {!r} (tensor cores enabled)'
             .format(precision))

    if args.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
        _log('[perf] cudnn.benchmark -> True')

    # Report the resulting state so every log records what was actually active.
    if precision != 'highest' or args.get('cudnn_benchmark', False):
        _log('[perf] matmul.allow_tf32={} cudnn.allow_tf32={} cudnn.benchmark={}'
             .format(torch.backends.cuda.matmul.allow_tf32,
                     torch.backends.cudnn.allow_tf32,
                     torch.backends.cudnn.benchmark))


def loader_kwargs(args, num_workers):
    """Build DataLoader kwargs, applying any ``dataloader`` overrides.

    ``num_workers`` is the value MiN would have used; it is overridden only if
    the config asks.  ``persistent_workers`` and ``prefetch_factor`` are only
    emitted when workers are actually in use, since PyTorch rejects them
    otherwise.
    """
    overrides = args.get('dataloader', {}) or {}

    workers = overrides.get('num_workers', num_workers)
    kwargs = {'num_workers': workers}

    if overrides.get('pin_memory', False):
        kwargs['pin_memory'] = True

    if workers > 0:
        if overrides.get('persistent_workers', False):
            kwargs['persistent_workers'] = True
        if 'prefetch_factor' in overrides:
            kwargs['prefetch_factor'] = overrides['prefetch_factor']

    return kwargs
