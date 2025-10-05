
# normal development
For normal development a jupyter-notebook might be an overkill, start the dev container with. (it will close after exiting)
```bash
docker run --name grap-dev-container -v "$(pwd):/home/fenics/shared" \ 
            -p 127.0.0.1:8888:8888 -p 127.0.0.1:8000:8000 -it graphnics
```

after having started it once you can reenter it with
```bash
#starts the container and drops you right in
docker start -ai grap-dev-container 
```

# jupyter-notebook development
Somtimes you might still need a jupyter-notebook,inside the dev container run this command
```bash
jupyter-notebook --ip=0.0.0.0  
```

Here are som other usefull docker commands
```bash
docker rm container-name # removes docker container
docker rmi container-image-name # removes a docker image
docker stop container-name # stops docker container
docker stats # lets you see what is running and how much resources are used

```

