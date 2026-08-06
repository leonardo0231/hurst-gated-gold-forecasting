# API

`GET /health` returns service health and the non-live trading mode.

`GET /status` returns the most recent execution report, or `not_executed` when the pipeline has not run.

`POST /validate-data` validates an in-memory OHLCV batch against the Phase 1 source contract. It does not persist the request and does not start training.

The API intentionally has no order, broker, credential, submission, model-retraining, or model-selection endpoint.

