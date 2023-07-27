import pytest

from streaming_wer import compute_streaming_wer


@pytest.mark.parametrize(
    ["tau", "expected"],
    [
        (-3, [20, 8]),
        (-2, [18, 8]),
        (-1, [17, 7]),
        (0, [15, 7]),
        (1, [13, 7]),
        (2, [11, 6]),
        (3, [9, 5]),
        (4, [7, 5]),
        (5, [6, 5]),
    ],
)
def test_streaming_wer(tau, expected):

    # Example (from Ehsan Variani):
    # This example calculates streaming character edit-distance. Note that we usually want word level edit-distance.

    # ref: kkkkkiiiiiitttttttttttteeeeeeeeennnnnnnnnn---iiiinnnnn----ttttthhhhhhheeeeeee--kkkkiiiiiittcccchhhhheeeeennnnnnn
    # reference timestamps (corresponding to the end of character)
    # t_r:     5    11  15      23       32        42 45  49   54  58   63     70     77 79 83    89 91 95  100  105    112
    # hyp: sssssssiiiiiitttttttttiiiiiiinnnnnnnnnnggggg--iiinn-----ttttthhhhhheeeeee-------kkkkkiiiitttttccccchhhhhheeeeeennnnn
    # t_h:       7    13   18  22     29        39   44 46 49 51 56   61    67    73     80   85  89   94   99   105   111  116
    # hypothesis timestamps (corresponding to the end of character)
    # c_del = 1
    # c_ins = 1
    # c_sub = 2
    # c_str = 1

    ref_texts = ["kitten in the kitchen", "kitten"]
    hyp_texts = ["sitting in the kitchen", "sitting"]

    ref = [[*ref_text] for ref_text in ref_texts]
    hyp = [[*hyp_text] for hyp_text in hyp_texts]

    # fmt: off
    ref_times = [[5, 11, 15, 23, 32, 42, 45, 49, 54, 58, 63,70, 77, 79, 83, 89, 91, 95, 100, 105, 112], [5, 11, 15, 23, 32, 42]]
    hyp_times = [[7, 13, 18, 22, 29, 39, 44, 46, 49, 51, 56, 61, 67, 73, 80, 85, 89, 94, 99, 105, 111, 116], [7, 13, 18, 22, 29, 39, 44]]
    # fmt: on

    all_tokens = sorted(set([*ref_texts[0]]) | set([*hyp_texts[0]]))
    int2sym = dict(enumerate(all_tokens))
    sym2int = {v: k for k, v in int2sym.items()}

    ref = [[sym2int[s] for s in r] for r in ref]
    hyp = [[sym2int[s] for s in h] for h in hyp]

    r = compute_streaming_wer(
        hyp,
        ref,
        hyp_times,
        ref_times,
        tau,
        ins_cost=1,
        del_cost=1,
        sub_cost=2,
        str_cost=1,
        reduction="none",
    )

    assert r.tolist() == expected
