import torch


class BinaryCrossEntropy:

    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
        self.eps = 1e-6

    def __call__(self, y, y_hat, mask=None, smoothness=0.65):
        # y : N1DHW
        # y_hat: N1DHW
        assert (y.size() == y_hat.size())
        # y_hat_sm = F.sigmoid(y_hat)
        y = y.float()
        p = y_hat
        t = y
        alpha = (1.0 - t.sum() / t.shape[0]).clamp(0.3, 0.7)
        pt = p * t + (1.0 - p) * (1.0 - t)  # pt = p if t > 0 else 1-p

        w = alpha * t + (1.0 - alpha) * (1.0 - t)  # w = alpha if t > 0 else 1-alpha

        ptc = pt.clamp(self.eps, 1. - self.eps)
        if mask is not None:
            nll_loss = -1.0 * (smoothness * torch.log(ptc) * w * mask
                               + torch.log(ptc) * w * (1.0 - mask))
        else:
            nll_loss = - smoothness * torch.log(ptc) * w
        bce_loss = nll_loss.sum() / w.sum()
        return bce_loss


def dice_coef(y, y_hat, smooth):
    y_hat_flat = y_hat.view(-1)
    y_flat = y.view(-1)
    intersection = (y_hat_flat * y_flat).sum()
    return (2. * intersection + smooth) / (y_flat.sum() + y_hat_flat.sum() + smooth)


class BinaryDice:

    def __init__(self, smooth):
        super(BinaryDice, self).__init__()
        self.smooth = smooth

    def __call__(self, y, y_hat, extra_args={}):
        return dice_coef(y, y_hat, self.smooth)
