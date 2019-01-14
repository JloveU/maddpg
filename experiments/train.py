import os
import sys
sys.path.append("..")
import argparse
import datetime
import maddpg.common.tf_util as U
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pyglet.gl
import tensorflow as tf
import tensorflow.contrib.layers as layers
import time
if "OPENAI_LOGDIR" not in os.environ: os.environ["OPENAI_LOGDIR"] = os.path.join("log", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if "OPENAI_LOG_FORMAT" not in os.environ: os.environ["OPENAI_LOG_FORMAT"] = "stdout,log,csv,tensorboard"
from baselines import logger
from baselines.run import parse_cmdline_kwargs
from maddpg.trainer.maddpg import MADDPGAgentTrainer


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=None, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, nargs="+", default=[64, 64], help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default=None, help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10000, help="save model once every time this many episodes are completed")
    parser.add_argument("--print-rate", type=int, default=1000, help="print training scalars once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default=None, help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save-render-images", action="store_true", default=False)
    parser.add_argument("--render-dir", type=str, default=None, help="directory in which render image should be saved")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default=None, help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default=None, help="directory where plot data is saved")
    args, unknown_args = parser.parse_known_args()
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.exp_name is None:
        args.exp_name = "experiment-{}".format(args.scenario)
    if args.save_dir is None:
        args.save_dir = os.path.join(logger.get_dir(), "checkpoints")
    if (args.render_dir is None) and (args.load_dir is not None):
        args.render_dir = args.load_dir + "-render"
    if args.benchmark_dir is None:
        args.benchmark_dir = os.path.join(logger.get_dir(), "benchmark_files")
    if args.plots_dir is None:
        args.plots_dir = os.path.join(logger.get_dir(), "learning_curves")

    if not args.display:
        os.makedirs(args.save_dir, exist_ok=True)
    if args.save_render_images:
        os.makedirs(args.render_dir, exist_ok=True)
    if args.benchmark:
        os.makedirs(args.benchmark_dir, exist_ok=True)
    if not args.display:
        os.makedirs(args.plots_dir, exist_ok=True)

    return args, extra_args


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        if isinstance(num_units, list):
            for num_units_ in num_units:
                out = layers.fully_connected(out, num_outputs=num_units_, activation_fn=tf.nn.relu)
        else:
            out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def make_env(scenario_name, arglist):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy == 'ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy == 'ddpg')))
    return trainers


def train(arglist, extra_args=None):
    tf_graph = tf.Graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(graph=tf_graph, config=tf_config):
        # Create environment
        env = make_env(arglist.scenario, arglist)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        if arglist.num_adversaries is None:
            arglist.num_adversaries = len([agent for agent in env.agents if (hasattr(agent, "adversary") and agent.adversary)])
        arglist.num_adversaries = min(env.n, arglist.num_adversaries)
        num_adversaries = arglist.num_adversaries
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        if os.environ.get("OUTPUT_GRAPH"):
            tf.summary.FileWriter(os.path.join(logger.get_dir(), "tb"), U.get_session().graph)

        # Load previous results, if necessary
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver(max_to_keep=None)
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
            # print("[action] " + ", ".join(["agent {i}: {action}".format(i=i, action=list(action_n[i])) for i in range(len(action_n))]))
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                if arglist.save_render_images:
                    input_file_name = os.path.join(arglist.render_dir, "image-episode_{}-step_%d.png".format(len(episode_rewards)))
                    output_file_name = os.path.join(arglist.render_dir, "video-episode_{}.mp4".format(len(episode_rewards)))
                    command = "ffmpeg -y -r 10 -i {} {}".format(input_file_name, output_file_name)
                    os.system(command)
                    print("Saved render video at {}".format(output_file_name))

                    for episode_step_ in range(episode_step):
                        file_name = os.path.join(arglist.render_dir, "image-episode_{}-step_{}.png".format(len(episode_rewards), episode_step_))
                        if os.path.exists(file_name):
                            os.remove(file_name)

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = os.path.join(arglist.benchmark_dir, 'benchmark.pkl')
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                if arglist.save_render_images:
                    images = env.render(mode="rgb_array")
                    image = images[0]
                    file_name = os.path.join(arglist.render_dir, "image-episode_{}-step_{}.png".format(len(episode_rewards), episode_step))
                    plt.imsave(file_name, image)
                    print("Saved render image at {}".format(file_name))
                else:
                    env.render(mode="human")
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(os.path.join(arglist.save_dir, "checkpoint-episode_{}".format(len(episode_rewards))), saver=saver)

            # print training scalars
            if terminal and ((len(episode_rewards) % arglist.print_rate == 0) or (len(episode_rewards) % arglist.save_rate == 0)):
                # print statement depends on whether or not there are adversaries
                logger.log("Time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                logger.logkv("steps", train_step)
                logger.logkv("episodes", len(episode_rewards))
                logger.logkv("mean_episode_reward", np.mean(episode_rewards[-arglist.save_rate:]))
                if num_adversaries == 0:
                    # print("[{}] steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    #     train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    pass
                else:
                    for agent_index in range(len(agent_rewards)):
                        logger.logkv("agent_{}_episode_reward".format(agent_index), np.mean(agent_rewards[agent_index][-arglist.save_rate:]))
                    # print("[{}] steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    #     train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                    #     [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                logger.logkv("time", round(time.time() - t_start, 3))
                logger.dumpkvs()
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = os.path.join(arglist.plots_dir, 'rewards.pkl')
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = os.path.join(arglist.plots_dir, 'average_rewards.pkl')
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


if __name__ == '__main__':
    logger.log("Command line: " + " ".join(sys.argv))
    arglist, extra_args = parse_args()
    train(arglist, extra_args)
