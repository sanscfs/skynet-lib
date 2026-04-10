# Custom Airflow image with skynet-lib preinstalled.
#
# Built in CI on self-hosted runners (tailnet access required for
# nexus.nexus.svc:8081 during pip install). Replaces the stock
# apache/airflow image in infra-apps/airflow/values.yaml — DAGs can
# then import `skynet_core`, `skynet_qdrant`, `skynet_embedding`,
# `skynet_matrix` directly instead of reimplementing lib primitives.
#
# Base tag is kept in sync with the Airflow Helm chart. Bump together
# with infra-apps/airflow/Chart.yaml + values.yaml.

FROM apache/airflow:3.1.8

USER airflow

COPY docker/airflow/requirements.txt /tmp/skynet-reqs.txt

RUN pip install --no-cache-dir \
        --extra-index-url http://nexus.nexus.svc:8081/repository/pypi-group/simple/ \
        --trusted-host nexus.nexus.svc \
        -r /tmp/skynet-reqs.txt \
    && rm /tmp/skynet-reqs.txt
