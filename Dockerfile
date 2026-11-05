# Neo4j 5 Community with APOC + Graph Data Science for the tutorial lessons.
FROM neo4j:5.26

# Plugins are fetched by the image's entrypoint on first start and cached in /plugins
# (a volume in docker-compose.yml, so later starts don't re-download them).
ENV NEO4J_PLUGINS='["apoc","graph-data-science"]' \
    NEO4J_dbms_security_procedures_unrestricted='gds.*,apoc.*' \
    NEO4J_dbms_security_procedures_allowlist='gds.*,apoc.*,db.*' \
    NEO4J_server_memory_heap_initial__size=512m \
    NEO4J_server_memory_heap_max__size=1G \
    NEO4J_server_memory_pagecache_size=512m

# Seed script for a fresh database (lessons/00-setup.md explains how to load it).
COPY seed/ /seed/

EXPOSE 7474 7687
