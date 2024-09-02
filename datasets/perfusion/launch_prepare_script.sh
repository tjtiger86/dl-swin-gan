#!/bin/bash

# Get directory that this script lives in
CODE_PATH=$(cd $(dirname $0); pwd -P)

# Get directory that data lives in (passed as first argument to this script)
DATA_PATH=$1

# Get directory of the Python Orchestra SDK
if [[ -z "${PYTHON_SDK_PATH}" ]]; then
  echo "You must set the Python Orchestra SDK path!"
  echo "i.e. export PYTHON_SDK_PATH=/path/to/orchestra-sdk-2.0-1.python"
  echo "Exiting script..."
  exit 1
else
  SDK_PATH="${PYTHON_SDK_PATH}"
fi

# Look to see if container exists already in docker
CONTAINER=python-sdk-tf
INSPECT="$(docker inspect --format='{{.Config.Image}}' $CONTAINER)"

# If not, then load it from the SDK directory
if [[ "$INSPECT" != sha* ]]
then
    docker load -i $SDK_PATH/python-sdk-tf.tar.gz
fi

docker run --rm -it \
       -v $DATA_PATH:/home/sdkuser/data \
       -v $CODE_PATH:/home/sdkuser/code \
       -v $SDK_PATH:/home/sdkuser/orchestra \
       -w /home/sdkuser \
       -p 8890:8890 \
       -u 0 \
       $CONTAINER python3 /home/sdkuser/code/prepare_dataset.py --directory /home/sdkuser/data

