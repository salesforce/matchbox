# before macro
def forward(x, h0):
    h = h0
    for i in range(x.size(1)):
        h = self.i2h(x[:, i]) + self.h2h(h)
    return h

# after macro
def forward(x, h0):
    h = h0
    def _loop_body_1(_i, _h):
        _tmp_1 = _b(self.i2h)(_b(x.__getitem__)(x, slice(None), _i))
        _h_1 = _b(_tmp_1.__add__)(_tmp_1, _b(self.h2h)(_h))
        return (_h_1,)
    h, = _for(_b(range)(_b(x.size)(x, 1)), _loop_body_1, (h,))
    return h
