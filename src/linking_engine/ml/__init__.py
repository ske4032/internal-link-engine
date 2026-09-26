"""Model training, MLflow tracking, and the promotion gate.

A model version transitions to ``Production`` only when its holdout NDCG@10
exceeds the version currently in ``Production``.
"""
