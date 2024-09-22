# EMLO

MLOps



## Session 2 [Docker]

- Train a CNN on MNIST from docker image
- Image size is restricted
- Model training resumes from last checkpoint
- Model is saved on the container



## Session 3 [Docker Compose]

- Using docker compose, create multiple services to train, test a CNN model on MNIST and perform inference
- Named volume is created and shared across all the services
- Data from each service is stored on the volume and can be accessed by other container
- Command to check content of the volume
- Request & Response services
- Basic swarm file to run multiple replicas of the same container



## Session 4 [Template Lightning Model]

- DevContainer used for the project