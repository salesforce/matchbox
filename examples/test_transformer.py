from matchbox.test_utils import mb_test

from transformer import LayerNorm, FeedForward, ResidualBlock, Attention, MultiHead

def test_LayerNorm():
    mb_test(LayerNorm(2),
            (4, (True, 3), (False, 2)))

def test_FeedForward():
    mb_test(FeedForward(2, 3, 0),
            (4, (True, 3), (False, 2)))

def test_ResidualBlock():
    mb_test(ResidualBlock(FeedForward(2, 3, 0), 2, 0),
            (4, (True, 3), (False, 2)))

def test_Attention():
    mb_test(Attention(2, 0, False),
            (4, (True, 3), (False, 2)), 0, 0)
    mb_test(Attention(2, 0, False),
            (4, (True, 3), (False, 2)), (4, (True, 3), (False, 2)), 1)
    mb_test(Attention(2, 0, True),
            (4, (True, 3), (False, 2)), 0, 0)

def test_MultiHead():
    mb_test(MultiHead(Attention(6, 0, False), 6, 6, 3),
            (4, (True, 3), (False, 6)), (4, (True, 3), (False, 6)), 1)
    mb_test(MultiHead(Attention(6, 0, True), 6, 6, 3),
            (4, (True, 3), (False, 6)), 0, 0)
