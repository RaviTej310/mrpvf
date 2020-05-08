#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import numpy as np

from arguments import get_args

# DQN
def dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    if config.control == False:
        config.fixed_policy = np.array([np.ones(13)*-1,
        [-1,0,2,2,2,2,-1,1,1,1,1,1,-1],
        [-1,0,0,0,0,0,-1,1,1,1,1,1,-1],
        [-1,0,0,0,0,0, 2,2,2,2,2,2,-1],
        [-1,0,0,0,0,0,-1,0,0,0,0,0,-1],
        [-1,0,0,0,0,0,-1,0,0,0,0,0,-1],
        [-1,-1,0,-1,-1,-1,-1,0,0,0,0,0,-1],
        [-1,3,0,2,2,2,-1,-1,-1,0,-1,-1,-1],
        [-1,0,0,0,0,0,-1, 1, 3,0, 2, 2,-1],
        [-1,0,0,0,0,0,-1, 1, 3,0, 2, 2,-1],
        [-1,0,0,0,0,0, 2, 2, 2,2, 2, 2,-1],
        [-1,0,0,0,0,0,-1, 0, 0,0, 0, 0,-1],
        np.ones(13)*-1])

        config.obs_map = np.array([np.ones(13)*-1,
        [-1,0,1,2,3,4,-1,5,6,7,8,9,-1],
        [-1,10,11,12,13,14,-1,15,16,17,18,19,-1],
        [-1,20,21,22,23,24,25,26,27,28,29,30,-1],
        [-1,31,32,33,34,35,-1,36,37,38,39,40,-1],
        [-1,41,42,43,44,45,-1,46,47,48,49,50,-1],
        [-1,-1,51,-1,-1,-1,-1,52,53,54,55,56,-1],
        [-1,57,58,59,60,61,-1,-1,-1,62,-1,-1,-1],
        [-1,63,64,65,66,67,-1,68,69,70,71,72,-1],
        [-1,73,74,75,76,77,-1,78,79,80,81,82,-1],
        [-1,83,84,85,86,87,88,89,90,91,92,93,-1],
        [-1,94,95,96,97,98,-1,99,100,101,102,103,-1],
        np.ones(13)*-1])

    config.task_fn = lambda: Task(config.game, seed = config.seed, goal_location = config.goal_location)
    config.eval_env = Task(config.game, seed = config.seed, goal_location = config.goal_location)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.01)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    # config.network_fn = lambda: DuelingNet(config.action_dim, FCBody(config.state_dim))
    config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=10)
    #config.replay_fn = lambda: AsyncReplay(memory_size=int(1e5), batch_size=30)

    if config.control == True:
        config.random_action_prob = LinearSchedule(1.0, 0.1, 1e6)
        config.max_steps = config.max_steps
    elif config.control == False:
        config.random_action_prob = LinearSchedule(0.1, 0.1, 1e6)
        config.max_steps = 200000
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 10000
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.async_actor = False
    run_steps(DQNAgent(config))


def dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(
        params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    # config.network_fn = lambda: DuelingNet(config.action_dim, NatureConvBody(in_channels=config.history_length))
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.batch_size = 32
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.history_length = 4
    # config.double_q = True
    config.double_q = False
    config.max_steps = int(2e7)
    run_steps(DQNAgent(config))


# QR DQN
def quantile_regression_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: QuantileNet(config.action_dim, config.num_quantiles, FCBody(config.state_dim))

    # config.replay_fn = lambda: Replay(memory_size=int(1e4), batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e4), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.num_quantiles = 20
    config.gradient_clip = 5
    config.sgd_update_frequency = 4
    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    run_steps(QuantileRegressionDQNAgent(config))


def quantile_regression_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00005, eps=0.01 / 32)
    config.network_fn = lambda: QuantileNet(config.action_dim, config.num_quantiles, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.num_quantiles = 200
    config.max_steps = int(2e7)
    run_steps(QuantileRegressionDQNAgent(config))


# C51
def categorical_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, FCBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)

    # config.replay_fn = lambda: Replay(memory_size=10000, batch_size=10)
    config.replay_fn = lambda: AsyncReplay(memory_size=10000, batch_size=10)

    config.discount = 0.99
    config.target_network_update_freq = 200
    config.exploration_steps = 100
    config.categorical_v_max = 100
    config.categorical_v_min = -100
    config.categorical_n_atoms = 50
    config.gradient_clip = 5
    config.sgd_update_frequency = 4

    config.eval_interval = int(5e3)
    config.max_steps = 1e5
    run_steps(CategoricalDQNAgent(config))


def categorical_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.optimizer_fn = lambda params: torch.optim.Adam(params, lr=0.00025, eps=0.01 / 32)
    config.network_fn = lambda: CategoricalNet(config.action_dim, config.categorical_n_atoms, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.01, 1e6)

    # config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=32)
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e6), batch_size=32)

    config.discount = 0.99
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.target_network_update_freq = 10000
    config.exploration_steps = 50000
    config.categorical_v_max = 10
    config.categorical_v_min = -10
    config.categorical_n_atoms = 51
    config.sgd_update_frequency = 4
    config.gradient_clip = 0.5
    config.max_steps = int(2e7)
    run_steps(CategoricalDQNAgent(config))


# A2C
def a2c_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, FCBody(config.state_dim, gate=F.tanh))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 0.5
    run_steps(A2CAgent(config))


def a2c_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: CategoricalActorCriticNet(config.state_dim, config.action_dim, NatureConvBody())
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    run_steps(A2CAgent(config))


def a2c_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=0.0007)
    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim), critic_body=FCBody(config.state_dim))
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 1.0
    config.entropy_weight = 0.01
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    run_steps(A2CAgent(config))


# N-Step DQN
def n_step_dqn_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 5
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim))
    config.random_action_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.gradient_clip = 5
    run_steps(NStepDQNAgent(config))


def n_step_dqn_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: VanillaNet(config.action_dim, NatureConvBody())
    config.random_action_prob = LinearSchedule(1.0, 0.05, 1e6)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    run_steps(NStepDQNAgent(config))


# Option-Critic
def option_critic_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: OptionCriticNet(FCBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    run_steps(OptionCriticAgent(config))


def option_critic_pixel(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.num_workers = 16
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=1e-4, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(), config.action_dim, num_options=4)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    run_steps(OptionCriticAgent(config))


# PPO
def ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: GaussianActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, 3e-4)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, 1e-3)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.max_steps = 3e6
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    run_steps(PPOAgent(config))


# DDPG
def ddpg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: DeterministicActorCriticNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body=TwoLayerFCBodyWithAction(
            config.state_dim, config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-4),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=64)
    config.discount = 0.99
    config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(
        size=(config.action_dim,), std=LinearSchedule(0.2))
    config.warm_up = int(1e4)
    config.target_network_mix = 1e-3
    run_steps(DDPGAgent(config))


# TD3
def td3_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()
    config.max_steps = int(1e6)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20

    config.network_fn = lambda: TD3Net(
        config.action_dim,
        actor_body_fn=lambda: FCBody(config.state_dim, (400, 300), gate=F.relu),
        critic_body_fn=lambda: FCBody(
            config.state_dim+config.action_dim, (400, 300), gate=F.relu),
        actor_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3),
        critic_opt_fn=lambda params: torch.optim.Adam(params, lr=1e-3))

    config.replay_fn = lambda: Replay(memory_size=int(1e6), batch_size=100)
    config.discount = 0.99
    config.random_process_fn = lambda: GaussianProcess(
        size=(config.action_dim,), std=LinearSchedule(0.1))
    config.td3_noise = 0.2
    config.td3_noise_clip = 0.5
    config.td3_delay = 2
    config.warm_up = int(1e4)
    config.target_network_mix = 5e-3
    run_steps(TD3Agent(config))


if __name__ == '__main__':
    args = get_args()
    print(args.transfer, args.seed)
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    seed = args.seed
    random_seed(seed=seed) 
    select_device(-1)
    # select_device(0)

    num_runs = 15   # DON'T CHANGE THIS - NEED TO KEEP THIS SAME TO ENSURE REPRODUCIBILITY. IF THIS IS CHANGED, DIFFERENT GOALS WILL BE CHOSEN
    goal_locations = np.random.randint(104,size=(2,num_runs))
    blocked_hallways = np.random.randint(4,size=(num_runs))

    iteration = args.run_index-1
    original_goal_location = goal_locations[0][iteration]
    transfer_goal_location = goal_locations[1][iteration] 

    algo = args.algo
    game = "FourRooms-v1"
    goal_location = original_goal_location
    new_goal_location = transfer_goal_location
    num_eigvals = args.num_eigvals
    transfer = args.transfer
    stochastic = args.stochastic
    if algo == 'mrpvf':
        decomp_type = args.decomp_type
    elif algo == 'pvf':
        decomp_type = ""

    # for policy evaluation
    if args.control == False:
        goal_location = 0
        tag = "_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"_"
        max_steps = 1500000
        dqn_feature(game=game, decomp_type = decomp_type, seed = seed, goal_location = goal_location, tag = tag, algo = algo, num_eigvals = num_eigvals, transfer = transfer, max_steps = max_steps, control = args.control)

    # for control
    elif args.control == True:
        if stochastic == False:
            tag = "_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"_"
        
            if transfer == False:   
                max_steps = 1500000
                dqn_feature(game=game, decomp_type = decomp_type, seed = seed, goal_location = goal_location, tag = tag, algo = algo, num_eigvals = num_eigvals, transfer = transfer, max_steps = max_steps, control = args.control)

            elif transfer == True:
                max_steps = 800000
                #load_weights_location = "log/"+algo+"_"+decomp_type+"_n="+str(num_eigvals)+"/"+"_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"__ntimesteps=1200000.pt"
                load_weights_location = "log/"+algo+"_"+decomp_type+"_n="+str(num_eigvals)+"_weight=2/"+"_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"__ntimesteps=1200000.pt"
                print(load_weights_location)
                tag = "_transfer_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"_"
                dqn_feature(game=game, decomp_type = decomp_type, seed = seed, goal_location = new_goal_location, tag = tag, algo = algo, num_eigvals = num_eigvals, transfer = transfer, max_steps = max_steps, load_weights_location = load_weights_location, control = args.control)

        elif stochastic == True:
            tag = "_stochastic_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"_"
        
            if transfer == False:   
                max_steps = 2000000
                dqn_feature(game=game, decomp_type = decomp_type, seed = seed, goal_location = goal_location, tag = tag, algo = algo, num_eigvals = num_eigvals, transfer = transfer, max_steps = max_steps, control = args.control)

            elif transfer == True:
                max_steps = 800000
                load_weights_location = "log/stochastic_"+algo+"_"+decomp_type+"_n="+str(num_eigvals)+"/"+"_stochastic_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"__ntimesteps=1200000.pt"
                print(load_weights_location)
                tag = "_stochastic_transfer_"+game+"_"+algo+"_"+decomp_type+"_seed="+str(seed)+"_run_iteration="+str(iteration)+"_num_eigvals="+str(num_eigvals)+"_"
                dqn_feature(game=game, decomp_type = decomp_type, seed = seed, goal_location = new_goal_location, tag = tag, algo = algo, num_eigvals = num_eigvals, transfer = transfer, max_steps = max_steps, load_weights_location = load_weights_location, control = args.control)


    # quantile_regression_dqn_feature(game=game)
    # categorical_dqn_feature(game=game)
    # a2c_feature(game=game)
    # n_step_dqn_feature(game=game)
    # option_critic_feature(game=game)

    # game = 'Hopper-v2'
    # a2c_continuous(game=game)
    # ppo_continuous(game=game)
    # ddpg_continuous(game=game)
    # td3_continuous(game=game)

    # game = 'BreakoutNoFrameskip-v4'
    # dqn_pixel(game=game)
    # quantile_regression_dqn_pixel(game=game)
    # categorical_dqn_pixel(game=game)
    # a2c_pixel(game=game)
    # n_step_dqn_pixel(game=game)
    # option_critic_pixel(game=game)
