"""An implementation of the MatchMaker model."""

from .base import UnimplementedModel

from chemicalx.models import Model
from chemicalx.data import DrugPairBatch
import torch
import torch.nn.functional as F  # noqa:N812

__all__ = [
    "MatchMaker",
]


class MatchMaker(Model):
    """An implementation of the MatchMaker model.

    .. seealso:: https://github.com/AstraZeneca/chemicalx/issues/23
    """

    def __init__(
        self,
        context_channels: int,
        drug_channels: int,
        input_hidden_channels: int = 32,
        middle_hidden_channels: int = 32,
        final_hidden_channels: int = 32,
        out_channels: int = 1,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.encoder = torch.nn.Linear(drug_channels + context_channels, input_hidden_channels)
        self.hidden_first = torch.nn.Linear(input_hidden_channels, middle_hidden_channels)
        self.hidden_second = torch.nn.Linear(middle_hidden_channels, middle_hidden_channels)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.scoring_head_first = torch.nn.Linear(2 * middle_hidden_channels, final_hidden_channels)
        self.scoring_head_second = torch.nn.Linear(final_hidden_channels, out_channels)

    def unpack(self, batch: DrugPairBatch):
        return (
            batch.context_features,
            batch.drug_features_left,
            batch.drug_features_right,
        )

    def forward(self, context_features, drug_features_left, drug_features_right):

        # drug_left
        hidden_left = torch.cat([context_features, drug_features_left], dim=1)
        hidden_left = self.encoder(hidden_left)
        hidden_left = F.relu(hidden_left)
        hidden_left = self.dropout(hidden_left)
        hidden_left = self.hidden_first(hidden_left)
        hidden_left = F.relu(hidden_left)
        hidden_left = self.dropout(hidden_left)
        hidden_left = self.hidden_second(hidden_left)

        # drug_right
        hidden_right = torch.cat([context_features, drug_features_right], dim=1)
        hidden_right = self.encoder(hidden_right)
        hidden_right = F.relu(hidden_left)
        hidden_right = self.dropout(hidden_right)
        hidden_right = self.hidden_first(hidden_right)
        hidden_right = F.relu(hidden_left)
        hidden_right = self.dropout(hidden_right)
        hidden_right = self.hidden_second(hidden_right)

        # both
        hidden_both = torch.cat([hidden_left, hidden_right], dim=1)

        hidden_both = self.scoring_head_first(hidden_both)
        hidden_both = F.relu(hidden_both)
        hidden_both = self.dropout(hidden_both)

        hidden_both = self.scoring_head_second(hidden_both)
        hidden_both = torch.sigmoid(hidden_both)
        return hidden_both

    def forward_with_thread(
        self,
        context_features: torch.FloatTensor,
        drug_features_left: torch.FloatTensor,
        drug_features_right: torch.FloatTensor,
    ):
        hidden_left = thread(
            torch.cat([context_features, drug_features_left], dim=1),
            [
                self.encoder,
                F.relu,
                self.dropout,
                self.hidden_first,
                F.relu,
                self.dropout,
                self.hidden_second,
            ],
        )

        hidden_right = thread(
            torch.cat([context_features, drug_features_right], dim=1),
            [
                self.encoder,
                F.relu,
                self.dropout,
                self.hidden_first,
                F.relu,
                self.dropout,
                self.hidden_second,
            ],
        )

        hidden_both = thread(
            torch.cat([hidden_left, hidden_right], dim=1),
            [
                self.scoring_head_first,
                F.relu,
                self.dropout,
                self.scoring_head_second,
                F.relu,
                self.dropout,
                torch.sigmoid,
            ],
        )

        return hidden_both


def thread(data, fns):
    acc = data
    for f in fns:
        acc = f(acc)
    return acc


def log(arg):
    print(arg)
    return arg
