"""FastAPI routers and dependencies.

The serving path. Contract ``api-excludes-ml`` forbids importing the training
tree (lightgbm, hdbscan, mlflow, sklearn, igraph, leidenalg) from here, so the
API image stays light.
"""
