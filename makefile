COMPUTE ?= gpu
RUNTIME ?= nvidia
IMAGE ?= 121565642659.dkr.ecr.us-east-1.amazonaws.com/tacotron2/docker-$(COMPUTE)
GIT_TAG ?= $(shell git log --oneline | head -n1 | awk '{print $$1}')
TAG ?= latest
#DCMD ?= docker run --rm -it --runtime=$(RUNTIME) -u $$(id -u):$$(id -g) -v $$(pwd)/../:/work -w /work/tacotron2 $(IMAGE):$(TAG)
DCMD             ?= docker run --rm -it --runtime=$(RUNTIME) -u $$(id -u):$$(id -g) -v $$(pwd)/../:/work -w /work/tacotron2 -e NVIDIA_VISIBLE_DEVICES=0 --shm-size=1g --ulimit memlock=-1 $(IMAGE):$(TAG)
TENSORBOARD_DCMD ?= docker run --rm -it --runtime=$(RUNTIME) -u $$(id -u):$$(id -g) -v $$(pwd)/../:/work -w /work/tacotron2 -p 6006:$(TENSORBOARD_PORT) $(IMAGE):$(TAG)

DATA_PATH ?= ../data
HOUR_STAMP ?= `date "+%Y%m%d_%H"`

.PHONY: docker
docker:
	docker build -t $(IMAGE) . -f Dockerfile.gpu
	docker tag $(IMAGE) $(IMAGE):$(GIT_TAG) 

docker-login:
	eval $$(aws ecr get-login --no-include-email --region us-east-1 | sed 's|https://||')

.PHONY: pull-docker
pull-docker:
	docker pull $(IMAGE):$(TAG)
	
train: train.py 
	$(DCMD) python3 $< --output_directory=outdir --log_directory=logdir --hparams=fp16_run=True

train-gpu: train.py
	$(DCMD) python3 -m multiproc $< --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True

train-nodocker: train.py 
	python3 $< --output_directory=outdir --log_directory=logdir

train-gpu-nodocker: train.py
	 python3 -m multiproc $< --output_directory=outdir --log_directory=logdir --hparams=distributed_run=True,fp16_run=True

TENSORBOARD_LOG_DIR ?= outdir/logdir
TENSORBOARD_PORT ?= 6006
tensorboard:
	$(TENSORBOARD_DCMD) tensorboard --logdir $(TENSORBOARD_LOG_DIR) --port $(TENSORBOARD_PORT)

watch-gpus:
	watch --interval 0.5 'nvidia-smi'

interact:
	$(DCMD) bash

interact-root:
	docker run --rm -it -v $$(pwd)/../:/work -w /work/tacotron2 $(IMAGE):$(TAG) bash
