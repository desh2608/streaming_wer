import torch
import streaming_wer_cuda as core
from pkg_resources import get_distribution

__version__ = get_distribution("streaming_wer").version


def streaming_levenshtein_distance(
    hypotheses,  # type: torch.IntTensor
    references,  # type: torch.IntTensor
    hypothesis_lengths,  # type: torch.IntTensor
    references_lengths,  # type: torch.IntTensor
    hypothesis_delays,  # type: torch.FloatTensor
    reference_delays,  # type: torch.FloatTensor
    threshold,  # type: float
    ins_cost=1,  # type: int
    del_cost=1,  # type: int
    sub_cost=2,  # type: int
    str_cost=1,  # type: int
):
    """Levenshtein streaming edit-distance for separated words or independent tokens.
    The difference from regular edit distance is that we also penalize correct tokens
    based on the delay cost. This is useful for streaming ASR
    where we want to penalize the delay of correct tokens.
    Return torch.ShortTensor (N, 4) with detail ins/del/sub/len statistics.

    Args:
      hypotheses (torch.Tensor): Tensor (N, H) where H is the maximum
        length of tokens from N hypotheses.
      references (torch.Tensor): Tensor (N, R) where R is the maximum
        length of tokens from N references.
      hypothesis_lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each hypothesis.
      references_lengths (torch.IntTensor): Tensor (N,) representing the
        number of tokens for each reference.
      hypothesis_offsets (torch.IntTensor): Tensor (N, H) representing the
        emission time step of each hypothesis token.
      reference_offsets (torch.IntTensor): Tensor (N, R) representing the
        emission time step of each reference token.
    """
    assert hypotheses.dim() == 2
    assert references.dim() == 2
    assert hypothesis_lengths.dim() == 1
    assert references_lengths.dim() == 1
    assert hypothesis_delays.dim() == 2
    assert reference_delays.dim() == 2
    assert hypotheses.size(0) == hypothesis_lengths.numel()
    assert references.size(0) == references_lengths.numel()
    assert hypothesis_lengths.numel() == references_lengths.numel()
    return core.streaming_levenshtein_distance(
        hypotheses,
        references,
        hypothesis_lengths,
        references_lengths,
        hypothesis_delays,
        reference_delays,
        threshold,
        ins_cost,
        del_cost,
        sub_cost,
        str_cost,
    )


def compute_streaming_wer(
    hs,
    rs,
    hd,
    rd,
    threshold,
    ins_cost=1,
    del_cost=1,
    sub_cost=2,
    str_cost=1,
    reduction="mean",
):
    """
    Args:
        hs (list[list[str]]): hypotheses
        rs (list[list[str]]): references
        hd (list[list[float]]): hypothesis delays
        rd (list[list[float]]): reference delays
        ins_cost (float): insertion cost
        del_cost (float): deletion cost
        sub_cost (float): substitution cost
        str_cost (float): streaming cost
        threshold (float): streaming threshold
    Returns:
        wer (torch.Tensor): streaming word error rate
    """
    # Get the lengths of the hypotheses and references.
    hyp_lengths = torch.tensor([len(h) for h in hs], dtype=torch.int, device="cuda")
    ref_lengths = torch.tensor([len(r) for r in rs], dtype=torch.int, device="cuda")

    # Convert the hypotheses and references into tensors.
    hyps = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(h, dtype=torch.int, device="cuda") for h in hs], batch_first=True
    )
    refs = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(r, dtype=torch.int, device="cuda") for r in rs], batch_first=True
    )
    hyp_delays = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(h, dtype=torch.float, device="cuda") for h in hd],
        batch_first=True,
    )
    ref_delays = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(r, dtype=torch.float, device="cuda") for r in rd],
        batch_first=True,
    )

    wer = streaming_levenshtein_distance(
        hyps,
        refs,
        hyp_lengths,
        ref_lengths,
        hyp_delays,
        ref_delays,
        float(threshold),
        ins_cost,
        del_cost,
        sub_cost,
        str_cost,
    ).float()
    if reduction == "mean":
        return wer.mean()
    elif reduction == "sum":
        return wer.sum()
    else:
        return wer
