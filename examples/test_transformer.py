from matchbox.test_utils import mb_test

from transformer import LayerNorm, FeedForward, ResidualBlock, Attention

def test_LayerNorm():
    mb_test(LayerNorm(2), (4, (True, 3), (False, 2)))

def test_FeedForward():
    mb_test(FeedForward(2, 3), (4, (True, 3), (False, 2)))

def test_ResidualBlock():
    mb_test(ResidualBlock(FeedForward(2, 3), 2, 0), (4, (True, 3), (False, 2)))

def test_Attention():
    mb_test(Attention())
