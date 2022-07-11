.PHONY: help check-required login build push build-and-push run run-sudo

# Arguments
# Document an argument on the line before it with the syntax '## VARNAME: description'

## NAME: (REQUIRED) The name for your project. Keep it short.
NAME ?= circuit-training
## USERNAME: (REQUIRED) Your username
USERNAME ?= mariasha
## REPO: Docker registry to push to
REPO = amr-registry.caas.intel.com/aipg
## TAG: Docker tag to use (useful for versioning) (default latest)
TAG ?= v2
## IMAGE: Full image name or image ID (default REPO/USERNAME-NAME:TAG). Useful for running a specific image ID
IMAGE ?= ${REPO}/${USERNAME}-${NAME}:${TAG}

## CONTEXT: Directory where the Dockerfile can be found (default PWD)
CONTEXT ?= $(shell pwd)
## FILE: Name of dockerfile (default Dockerfile)
FILE ?= ubuntu_circuit_training

## BUILD_ARGUMENTS: Additional --build-args for docker build
BUILD_ARGUMENTS ?=

## RUN_ARGUMENTS: Additional --env for docker run
RUN_ARGUMENTS ?=
## COMMAND: Command to run inside docker container when calling docker run (default /bin/bash)
COMMAND ?= /bin/bash

check-required:
ifeq ($(strip $(NAME)),)
	$(error NAME is undefined. Edit it in the file or set it as an environment variable)
endif
ifeq ($(strip $(USERNAME)),)
	$(error USERNAME is undefined. Edit it in the file or set it as an environment variable)
endif

# Proxy variables
ifdef http_proxy
    BUILD_ARGUMENTS+=--build-arg http_proxy=$(http_proxy)
    RUN_ARGUMENTS+=--env http_proxy=$(http_proxy)
endif
ifdef https_proxy
    BUILD_ARGUMENTS+=--build-arg https_proxy=$(https_proxy)
    RUN_ARGUMENTS+=--env https_proxy=$(https_proxy)
endif

# Document a command by adding "## <description>" after the rule
help:
	@echo "Docker image helper"
	@echo "====================="
	@echo "Build, run and push a docker image for use within the cluster"
	@echo
	@echo "Commands:"
	@grep -E '^[a-zA-Z0-9_%/-]+:.*?##.*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?##"}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Arguments:"
	@echo "Set these as environment variables or at the top of the Makefile"
	@grep -E '^##\s*[a-zA-Z0-9_%/-]+:.*$$' $(MAKEFILE_LIST) | sed 's/## *//' | awk 'BEGIN {FS = ":"}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Examples:"
	@echo
	@echo "Build and push using the details specified in the Makefile"
	@echo "  >> make build-and-push"
	@echo
	@echo "Build and push a Dockerfile located at /tmp/docker/Dockerfile.base, naming it <USERNAME>-test-docker"
	@echo "  >> FILE=Dockerfile.base CONTEXT=/tmp/docker NAME=test-docker make build-and-push"
	@echo
	@echo "Enter a shell at the docker ID 8cda75195d2c"
	@echo "  >> IMAGE=8cda75195d2c make run"
	@echo


# Use nvidia-docker if it is available
DOCKER := $(shell command -v nvidia-docker 2> /dev/null)
ifndef DOCKER
    DOCKER = docker
endif

name:  ## Return the full Docker image name
	@echo "${IMAGE}"

# Docker login
# ------------

login: ## Login to repo
	${DOCKER} login ${REPO}


# Docker build
# ------------

build: check-required ## Build the image from a Dockerfile
	@echo "Building docker image: ${IMAGE}"
	${DOCKER} build -f=${CONTEXT}/${FILE} -t=${IMAGE} ${BUILD_ARGUMENTS} ${CONTEXT}

# Docker push
# -----------

push: check-required login  ## Push the image
	@echo "Pushing ${IMAGE}"
	${DOCKER} push ${IMAGE}

build-and-push: build push  ## Build the image first, then push

# Docker run
# ----------

run: ## Run a bash shell in the container
	docker run -it --rm ${RUN_ARGUMENTS} ${IMAGE} ${COMMAND}

run-sudo: ## Run a bash shell as root in the container
	docker run -it --rm -u root $(RUN_ARGUMENTS) ${IMAGE} ${COMMAND}

