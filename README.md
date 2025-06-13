# sentence_tagging

docker compose -f container/docker-compose.yml build
docker compose -f container/docker-compose.yml up api


docker compose -f container/docker-compose.yml run --rm -it api /bin/bash
docker compose -f container/docker-compose.yml exec -it api /bin/bash

http://localhost:8000/docs#/