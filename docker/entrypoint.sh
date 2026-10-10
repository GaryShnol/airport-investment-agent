#!/bin/sh
# Prepare the DB (download the seed if needed, then the freshness check), then serve the app.
set -e
python -m src.bootstrap
exec streamlit run app.py --server.port "${APP_PORT}" --server.address 0.0.0.0
