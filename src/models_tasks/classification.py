"""In this python module there are the models needed for the projects."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy


# ==================================================================
#
#                         MODELS DEFINITION
#
# ==================================================================
class Classifier(pl.LightningModule):
    """An implementation of a classifier using a MLP architecture in pytorch.

    Args:
        input_dim : int
            The input dimension.
        num_classes : int
            The number of classes. Default 20.
        hidden_dim : int
            The hidden layer dimension. Default 10.
        lr : float
            The learning rate. Default 1e-2.
        momentum : float
            How much momentum to apply. Default 0.9.
        nesterov : bool
            If set to True use nesterov type of momentum. Default True.
        max_lr : float
            Maximum learning rate for the scheduler. Default 1..

    Attributes:
        self.hparams["<name-of-argument>"]:
            ex. self.hparams["input_dim"] is where the 'input_dim' argument is stored.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 20,
        hidden_dim: int = 10,
        lr: float = 1e-2,
        momentum: float = 0.9,
        nesterov: bool = True,
        max_lr: float = 1.0,
    ):
        super().__init__()

        # Log the hyperparameters.
        self.save_hyperparameters()

        self.accuracy = MulticlassAccuracy(
            num_classes=self.hparams['num_classes']
        )

        # Example input
        self.example_input_array = torch.randn(self.hparams['input_dim'])

        self.model = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.hparams['input_dim']),
            nn.Linear(self.hparams['input_dim'], self.hparams['hidden_dim']),
            nn.Tanh(),
            nn.LayerNorm(normalized_shape=self.hparams['hidden_dim']),
            nn.Linear(self.hparams['hidden_dim'], self.hparams['num_classes']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The forward pass of the Classifier.

        Args:
            x : torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output of the MLP.
        """
        x = nn.functional.normalize(x, p=2, dim=-1)
        return self.model(x)

    def configure_optimizers(self) -> dict[str, object]:
        """Define the optimizer: Stochastic Gradient Descent.

        Returns:
            dict[str, object]
                The optimizer and scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])
        return {
            'optimizer': optimizer,
        }

    def loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A convenient method to get the loss on a batch.

        Args:
            x : torch.Tensor
                The input tensor.
            y : torch.Tensor
                The original output tensor.

        Returns:
            (logits, loss) : tuple[torch.Tensor, torch.Tensor]
                The output of the MLP and the loss.
        """
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        return logits, loss

    def _shared_eval(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A common step performed in the test and validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            prefix : str
                The step type for logging purposes.

        Returns:
            (logits, loss) : tuple[torch.Tensor, torch.Tensor]
                The tuple with the output of the network and the epoch loss.
        """
        x, y = batch
        logits, loss = self.loss(x, y)

        # Getting the predictions
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log(f'{prefix}/loss_epoch', loss, on_step=False, on_epoch=True)
        self.log(f'{prefix}/acc_epoch', acc, on_step=False, on_epoch=True)

        return preds, loss

    def training_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """The training step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            loss : torch.Tensor
                The epoch loss.
        """
        x, y = batch
        logits, loss = self.loss(x, y)

        # Getting the predictions
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('train/loss', loss, on_epoch=True)
        self.log('train/acc', acc, on_epoch=True)

        return loss

    def test_step(self, batch: list[torch.Tensor], batch_idx: int) -> None:
        """The test step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            None
        """
        _ = self._shared_eval(batch, batch_idx, 'test')
        return None

    def validation_step(
        self, batch: list[torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """The validation step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.

        Returns:
            preds : torch.Tensor
                The output of the network.
        """
        preds, _ = self._shared_eval(batch, batch_idx, 'valid')
        return preds

    def predict_step(
        self, batch: list[torch.Tensor], batch_idx: int, dataloader_idx=0
    ) -> torch.Tensor:
        """The predict step.

        Args:
            batch : list[torch.Tensor]
                The current batch.
            batch_idx : int
                The batch index.
            dataloader_idx : int
                The dataloader idx.

        Returns:
            preds : torch.Tensor
                The output of the network.
        """
        x = batch[0]

        logits = self(x)
        preds = torch.argmax(logits, dim=1)

        return preds

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class probabilities for each observation.

        Args:
            x : torch.Tensor
                Input tensor.

        Returns:
            torch.Tensor
                Class probabilities of shape (batch_size, num_classes)
        """
        logits = self(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def entropy(
        self, x: torch.Tensor, normalized: bool = True
    ) -> torch.Tensor:
        """Compute predictive entropy per observation.

        Args:
            x : torch.Tensor
                Input tensor.
            normalized : bool
                If True, normalize by log(num_classes) to range [0, 1].

        Returns:
            torch.Tensor
                Entropy per observation (shape: [batch_size])
        """
        probs = self.predict_proba(x)
        eps = 1e-12
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)

        if normalized:
            entropy /= torch.log(
                torch.tensor(
                    self.hparams['num_classes'],
                    device=entropy.device,
                    dtype=entropy.dtype,
                )
            )
        return entropy

    def precision_weight(
        self, x: torch.Tensor, normalized: bool = True
    ) -> torch.Tensor:
        """Compute a precision-like confidence weight based on entropy.

        Args:
            x : torch.Tensor
                Input tensor.
            normalized : bool
                Whether to normalize entropy before inversion.

        Returns:
            torch.Tensor
                Precision weights (inverse of entropy), shape: [batch_size]
        """
        ent = self.entropy(x, normalized=normalized)
        weights = 1.0 / (ent + 1e-6)
        weights = weights / weights.sum()  # normalize so weights sum to 1
        return weights


def main() -> None:
    """The main script loop in which we perform some sanity tests."""

    print('Start performing sanity tests...')

    # Variables definition
    input_dim: int = 16
    num_classes: int = 2
    hidden_dim: int = 10
    hidden_size: int = 4

    data = torch.randn(10, input_dim)

    print()
    print('Test for Classifier...', end='\t')
    mlp = Classifier(input_dim, num_classes, hidden_dim, hidden_size)
    mlp(data)
    print('[Passed]')

    return None


if __name__ == '__main__':
    main()
