import torch

from riichi_analysis_engine.semantic_v16 import V16Input, _tile_neighbours


def test_directional_tile_edges_respect_suits_honors_and_red_fives():
    left, right, red = _tile_neighbours()
    assert (left[0], right[0], red[0]) == (-1, 1, -1)
    assert (left[8], right[8], red[8]) == (7, -1, -1)
    assert (left[9], right[9], red[9]) == (-1, 10, -1)
    assert (left[4], right[4], red[4]) == (3, 5, 34)
    assert (left[34], right[34], red[34]) == (3, 5, 4)
    assert (left[13], right[13], red[13]) == (12, 14, 35)
    assert (left[22], right[22], red[22]) == (21, 23, 36)
    assert (left[27:34] == -1).all()
    assert (right[27:34] == -1).all()
    assert (red[27:34] == -1).all()


def test_public_call_consumed_order_cannot_change_learned_event_encoding():
    torch.manual_seed(16)
    encoder = V16Input(256)
    event = torch.tensor([5, 2, 1, 6, 5, 35, 0, 0, 0], dtype=torch.uint8)[None, None]
    reordered = event.clone()
    reordered[..., 4], reordered[..., 5] = event[..., 5], event[..., 4]
    torch.testing.assert_close(
        encoder.encode_event_tiles(event.long()),
        encoder.encode_event_tiles(reordered.long()),
    )
