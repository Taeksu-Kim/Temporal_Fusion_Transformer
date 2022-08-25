import torch

class QuantileLossCalculator():
    """Computes the combined quantile loss for prespecified quantiles.
    Attributes:
      quantiles: Quantiles to compute losses
    """
    def __init__(self, config):
        """Initializes computer with quantiles for loss calculations.
        Args:
          quantiles: Quantiles to use for computations.
        """
        self.config = config
        self.quantiles = self.config.quantiles

    def quantile_loss(self, a, b):
        """Returns quantile loss for specified quantiles.
        Args:
          a: Targets
          b: Predictions
        """
        quantiles_used = set(self.quantiles)

        loss = 0.
        for i, quantile in enumerate(self.quantiles):
          if quantile in quantiles_used:
            loss += tensorflow_quantile_loss(
                a[Ellipsis, self.config.output_size * i:self.config.output_size * (i + 1)],
                b[Ellipsis, self.config.output_size * i:self.config.output_size * (i + 1)], quantile)
        return loss


# Loss functions.
def tensorflow_quantile_loss(y, y_pred, quantile):
    """Computes quantile loss for tensorflow.
    Standard quantile loss as defined in the "Training Procedure" section of
    the main TFT paper
    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)
    Returns:
      Tensor for quantile loss.
    """

    # Checks quantile
    if quantile < 0 or quantile > 1:
        raise ValueError(
            'Illegal quantile value={}! Values should be between 0 and 1.'.format(
                quantile))

    prediction_underflow = y - y_pred
    q_loss = quantile * torch.maximum(prediction_underflow, torch.tensor(0.)) + (
        1. - quantile) * torch.maximum(-prediction_underflow, torch.tensor(0.))

    return torch.sum(q_loss, dim=-1)