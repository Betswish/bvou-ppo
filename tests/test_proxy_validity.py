from beippo.proxy_validity import (
    _top1_hit,
    _topk_overlap_fraction,
    _topk_union,
    spearman_correlation,
)


def test_spearman_correlation_perfect_positive():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [10.0, 20.0, 30.0, 40.0]
    assert spearman_correlation(x, y) == 1.0


def test_topk_union_collects_from_all_proxies():
    score_maps = {
        "a": {0: 1.0, 1: 5.0, 2: 3.0},
        "b": {2: 10.0, 3: 9.0, 4: 8.0},
    }
    assert _topk_union(score_maps, top_k=2) == [1, 2, 3]


def test_topk_overlap_fraction():
    scores = {0: 10.0, 1: 9.0, 2: 1.0, 3: 0.5}
    gains = {0: 7.0, 2: 6.0, 1: 5.0, 3: 1.0}
    assert _topk_overlap_fraction(scores, gains, 2) == 0.5


def test_top1_hit():
    scores = {0: 0.1, 1: 0.9}
    gains = {0: 0.2, 1: 0.8}
    assert _top1_hit(scores, gains) == 1.0
