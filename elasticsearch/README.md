# Part 3: Elasticsearch

## Table of Contents
- [Setup](#setup)

## Setup
1. First we need to run an Elasticsearch instance. It is recommended to use the [Elasticsearch Docker image](https://hub.docker.com/_/elasticsearch).
```bash
docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" -e "xpack.security.enabled=false" elasticsearch:8.3.2
```
2. Now you should set `elastic_address' value in the `config.json` file.
3. Start the application with the following command.
```bash
python3 elasticsearchretrieval.py
```
