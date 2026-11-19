"""Uneven pipeline splits: per-stage decoder layer counts -> Megatron layout string.

A split is a tuple of decoder-layer counts, one per pipeline stage. With virtual
pipelining there is one entry per model chunk, in Megatron's order: entry i is
VPP chunk i // PP on PP rank i % PP. The Megatron side is
TransformerConfig.pipeline_model_parallel_layout (present in megatron-core 0.13.1 and
0.16.0), whose offsets drive get_transformer_layer_offset / get_num_layers_to_build.
"""


def parse_stage_layers(text):
    """Parse "7,9,9,7" into (7, 9, 9, 7); None or "" means the even split."""
    if text is None or text == "":
        return None
    try:
        counts = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise ValueError(f"pipeline stage layers must be comma-separated integers, got {text!r}")
    return counts


def validate_stage_layers(counts, num_layers, pipeline_parallel_size, virtual_pipeline_size=None):
    """Check a split against the model and parallel sizes; returns the counts as a tuple."""
    counts = tuple(counts)
    stages = pipeline_parallel_size * (virtual_pipeline_size or 1)
    if len(counts) != stages:
        raise ValueError(f"pipeline stage layers has {len(counts)} entries; PP * VPP = {stages}")
    if any(type(c) is not int or c < 1 for c in counts):
        raise ValueError("every pipeline stage must hold at least one decoder layer")
    if sum(counts) != num_layers:
        raise ValueError(f"pipeline stage layers sum to {sum(counts)}, but num_layers is {num_layers}")
    return counts


def layout_string(counts, mtp_num_layers=0):
    """Megatron layout: embedding on the first stage, loss (and any MTP layers) on the last."""
    stages = [f"t*{c}" for c in counts]
    stages[0] = "E" + stages[0]
    stages[-1] = stages[-1] + "m" * (mtp_num_layers or 0) + "L"
    return "|".join(stages)


def stage_ranges(counts):
    """(start, width) of each stage's contiguous global decoder layers."""
    ranges, start = [], 0
    for count in counts:
        ranges.append((start, count))
        start += count
    return tuple(ranges)
