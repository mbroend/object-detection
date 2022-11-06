# Instructions

## Requirements

* NVIDIA GPU with the latest driver installed
* docker / nvidia-docker

This build has been tested with Nvidia Drivers 526.47 and CUDA 11.2 on a Windows 10 machine using WSL2 backend.
Please update the base image if you plan on using other versions of CUDA.

## Build
Build the image with:
```
docker build -t project-dev -f Dockerfile .
```

Create a container with:
(!!! Replace <absolute path of git repo> !!!  e.g. ~/projects/udacity/dsnd/object-detection/nd013-c1-vision-starter:/app/project/ !!!)
```
docker run --gpus all -v <absolute path of git repo>:/app/project/ -p 8888:8888 -p 6006:6006 -ti project-dev bash 
```

and any other flag you find useful to your system (eg, `--shm-size` is usefull for downloading and processing the files). 


## Set up

Once in container, you will need to auth using:
```
gcloud auth login
```
