# Deploy in Swarm

```
docker swarm init
```

```
docker stack deploy -c docker-compose.yml mnist_server
```

```
docker stack ps mnist_server
docker service ls
```

```
docker stack rm mnist_server
```