# TreeQN and ATreeC: Differentiable Tree Planning for Deep Reinforcement Learning
Code of our ICLR 2018 [paper](https://arxiv.org/abs/1710.11417).

## Requirements

The code can be run in a docker container as described below. The dockerfile template in `docker/Dockerfile.cuda.template` lists all requirements that may be needed to set up a non-dockerised environment. The core requirements are pytorch, gym, and sacred.

## Sacred

The configuration and logging is handled by [Sacred](https://sacred.readthedocs.io/en/latest/). Results are stored by the FileStorageObserver as .json's in `results/`. We recommend a MongoObserver (requires pymongo) to organise larger numbers of experiments.

## Options
Valid configuration options are documented in `conf/default.yaml`.
The default settings correspond to our Atari experiments on Seaquest with TreeQN, depth 2.

## Running

To run a default setup with the configuration specified in `conf/default.yaml`, simply execute:
```
python treeqn/nstep_run.py
```

Further parameters can be specified using `with`:
```
python treeqn/nstep_run.py with env_id=Qbert architecture=dqn
```

Configuration files can also be used. Our box-pushing experiment defaults are given in `conf/push.yaml`:
```
python treeqn/nstep_run.py with config=./conf/push.yaml
```

If you have [Docker](https://docs.docker.com/) installed, you can build a docker image tagged `treeqn` with:
```
cd docker
./build.sh
cd ..
```

To run an experiment in a detached docker container named `treeqn-$GPU_ID`, use:
```
./docker/run.sh $GPU_ID python treeqn/nstep_run.py
```

## Citation
```
@inproceedings{farquhar2018treeqn,
  title={TreeQN and ATreeC: Differentiable Tree-Structured Models for Deep Reinforcement Learning},
  author={Farquhar, Gregory and Rockt{\"a}schel, Tim and Igl, Maximilian and Whiteson, Shimon},
  booktitle={ICLR 2018},
  year={2018}
}
```
