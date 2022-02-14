################
# Question 1.2 #
################

### For the Ant task ###

- for values of SEED between 0 and 6 inclusive, run:
$ python run_hw1.py env.exp_name='bc' logging.random_seed=SEED


### For the Walker2d task ###

- for values of SEED between 0 and 6 inclusive, run:
$ python run_hw1.py env.exp_name='bc' env.env_name='Walker2d-v2' env.expert_data='./ift6163/expert_data/expert_data_Walker2d-v2.pkl' env.expert_policy_file='./ift6163/policies/experts/Walker2d.pkl' logging.random_seed=SEED


################
# Question 1.3 #
################

- for values of LENGTH between 100 and 1000 inclusive (with increments of 100), run:
$ python run_hw1.py env.exp_name='bc' env.max_episode_length=LENGTH


################
# Question 2 #
################

### For the Ant task ###

- choose a value of SEED, then run:
$ python run_hw1.py env.exp_name='dagger' alg.do_dagger=true alg.n_iter=10 logging.random_seed=SEED


### For the Walker2d task ###

- choose a value of SEED, then run:
$ python run_hw1.py env.exp_name='dagger' alg.do_dagger=true alg.n_iter=10 env.env_name='Walker2d-v2' env.expert_data='./ift6163/expert_data/expert_data_Walker2d-v2.pkl' env.expert_policy_file='./ift6163/policies/experts/Walker2d.pkl' logging.random_seed=SEED

###############
# config.yaml #
###############

For the default hyper parameter values, see the file at ./conf/config.yaml

For convenience, the values are reproduced below:

env: 
  expert_policy_file: ./ift6163/policies/experts/Ant.pkl # Relative to where you're running this script from
  expert_data: ./ift6163/expert_data/expert_data_Ant-v2.pkl  # Relative to where you're running this script from
  exp_name: "bob"
  env_name: Ant-v2 # choices are [Ant-v2, Humanoid-v2, Walker2d-v2, HalfCheetah-v2, Hopper-v2]
  max_episode_length: 1000
  render: false

alg:
  num_rollouts: 5
  do_dagger: false
  num_agent_train_steps_per_iter: 1000 # number of gradient steps for training policy (per iter in n_iter)
  n_iter: 1
  batch_size: 1000 # training data collected (in the env) during each iteration
  eval_batch_size: 5000 # eval data collected (in the env) for logging metrics
  train_batch_size: 100 # number of sampled data points to be used per gradient/train step
  n_layers: 2 # Network depths
  network_width: 64 # The width of the network layers
  learning_rate: 5e-3 # THe learning rate for BC
  max_replay_buffer_size: 100000 ## Size of the replay buffer 1e5
  use_gpu: False
  which_gpu: 0 # The index for the GPU (the computer you use may have more than one)
  discrete: False
  ac_dim: 0 ## This will be overridden in the code
  ob_dim: 0 ## This will be overridden in the code

logging:
  video_log_freq: 1000 # How often to generate a video to log/
  scalar_log_freq: 1 # How often to log training information and run evaluation during training.
  save_params: true # Should the parameters given to the script be saved? (Always...)
  random_seed: 0


