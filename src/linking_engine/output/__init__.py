"""The tenant's served output in Mongo: the recommendations stage writes one run at a time,
the output API reads the latest complete run. Neither side needs the graph or the training
tree."""
