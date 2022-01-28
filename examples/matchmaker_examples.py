"""Example with DeepSynergy."""

from chemicalx import pipeline
from chemicalx.data import DrugCombDB
from chemicalx.models import DeepSynergy, MatchMaker
import torch
import numpy as np


dataset = DrugCombDB()

ctx_chs = dataset.context_channels

drug_chs = dataset.drug_channels


# model = DeepSynergy(context_channels=ctx_chs, drug_channels=drug_chs)

model = MatchMaker(ctx_chs, drug_chs)
ctx_features = torch.FloatTensor(np.random.uniform(0, 1, (1000, ctx_chs)))
drug_features_left = torch.FloatTensor(np.random.uniform(0, 1, (1000, drug_chs)))
drug_features_right = torch.FloatTensor(np.random.uniform(0, 1, (1000, drug_chs)))

model.forward(ctx_features, drug_features_left, drug_features_right)


results = pipeline(
    dataset=dataset,
    model=model,
    batch_size=5120,
    epochs=100,
    context_features=True,
    drug_features=True,
    drug_molecules=False,
)

results.summarize()
