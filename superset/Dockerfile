FROM apache/superset:latest

# Switch to root to install packages
USER root

# Install DuckDB and the SQLAlchemy driver
RUN pip install "duckdb==0.10.3" "duckdb-engine==0.13.6" "SQLAlchemy==1.4.51"

# Switch back to the superset user
USER superset