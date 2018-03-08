( echo '### DO NOT EDIT DIRECTLY, SEE Dockerfile.template ###'; sed -e "s/<<UID>>/${UID}/" < Dockerfile.cuda.template ) > Dockerfile
if hash nvidia-docker 2>/dev/null; then
    nvidia-docker build -t treeqn .
else
    docker build -t treeqn .
fi
