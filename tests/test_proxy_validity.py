from beippo.proxy_validity import _topk_union, spearman_correlation


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
