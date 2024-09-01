#! /bin/bash
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(id -ng)
CONTAINER_NAME=onda_htps #change 'onda' to your name
PROJECT_NAME=onda_htps #change 'onda' to your name

cat << EOF > .env
USER_ID=$USER_ID
GROUP_ID=$GROUP_ID
USERNAME=$USERNAME
CONTAINER_NAME=$CONTAINER_NAME
PROJECT_NAME=$PROJECT_NAME
EOF

docker-compose -p $PROJECT_NAME up -d --build
