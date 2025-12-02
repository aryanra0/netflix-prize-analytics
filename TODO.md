# TODOs & Future Ideas

Here are some things I didn't get to, but would be cool to add:

- [ ] **Hyperparameter Tuning**: Use `GridSearchCV` or `Optuna` to find the best params for SVD and XGBoost.
- [ ] **Real-time Inference**: The current API simulates predictions for speed. Hook it up to the real `.pkl` models for production use.
- [ ] **Dockerize**: Wrap the whole thing in a Docker container so it runs anywhere without Java/Python version headaches.
- [ ] **More Features**: Add "Genre" or "Tag" features if I can find external metadata for the movies.

## Risky / Skipped Items
- **Full Dataset Training**:I currently sample 5M rows to save RAM. Training on the full 100M ratings requires a cluster or a beefy machine (32GB+ RAM).

## Known Issues
- **Unit Tests**: `pytest` fails with an `ImportError` in `unittest.loader`. This appears to be a local environment issue with the Python installation, as the pipeline scripts (`run.sh`) execute correctly.
