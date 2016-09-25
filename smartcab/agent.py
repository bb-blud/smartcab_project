import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

number_trials = 100

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, policy, alpha, gamma, no_plot):
        super(LearningAgent, self).__init__(env)                   # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'                                         # override color
        self.planner = RoutePlanner(self.env, self)                # simple route planner to get next_waypoint
        self.policy = policy

        # State descriptors
        self.actions = self.env.valid_actions
        self.lights  = ['green','red']
        
        # For a global tally of events over the n trials
        self.total_time = 0
        self.trial = -1
        self.no_plot = no_plot   # activate plots or not
        self.bad_actions  = [0 for trial in range(number_trials)]  # bad actions performed in a given trial
        self.out_of_times = [0 for trial in range(number_trials)]  # trials that the agent ran out of time

        
        # For Q learning implementation
        self.gamma = gamma
        self.alpha = alpha

        self.Q = {
                  (action, 'green', oncoming, waypoint) : 1 \
                  for action   in self.actions              \
                  for oncoming in self.actions              \
                  for waypoint in self.actions[1:]            ## waypoint is only None when target is reached
            }

        red_Q =  {
                  (action, 'red', left, waypoint) : 1 \
                  for action   in self.actions        \
                  for left     in self.actions        \
                  for waypoint in self.actions[1:]
            }

        self.Q.update(red_Q)                                  ## combining the two main learning scenarios into dictionary Q

    def reset(self, destination=None):
        self.planner.route_to(destination)

        # Reset previous state and increment tally
        self.prev_action_state = None
        self.prev_reward = 0
        self.trial += 1                                       ## update the trial count

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()     ## from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        wp = self.next_waypoint
        
        # # Update state
        if inputs['light'] == 'green':
            state = (inputs['light'], inputs['oncoming'], wp)
        else:
            state = (inputs['light'], inputs[  'left'  ], wp)

        # # Select action according to policy
        Q_action = self.max_action(state)

        action  = {
                   "random"        : random.choice(self.actions[1:]),
                   "reckless"      : wp,
                   "semi_reckless" : self.semi_reckless(Q_action, state),
                   "Q_learning"    : Q_action } [self.policy]    ## Dictionary of different policies for comparison


        # # Execute action and get reward
        reward = self.env.act(self, action)

        # # Learn policy based on state, action, reward
        if not self.prev_action_state:              ## First update previous state is also current state
            self.prev_action_state = (action,) + state
            self.prev_reward = reward

        ## Q Update ##
        alpha = self.alpha                          ## Learning rate

        V = self.Q[ self.prev_action_state ]
        X = self.prev_reward + self.gamma * self.Q[ (action,) + state]

        self.Q[ self.prev_action_state ] =  (1-alpha) * V + alpha * X
        ##

        self.prev_action_state = (action,) + state  ## current state will be previous state on next update
        self.prev_reward = reward

        # # Tally bad actions
        self.tally(reward, t)

        # print "LearningAgent.update(): deadline = {}, inputs = {}".format(deadline, inputs)
        # print "wp = {}, action = {}, reward = {}, Q = {}".format(wp, action, reward,V)  # [debug]
        
    def max_action(self, state):
        actions_and_vals = {action : self.Q[ (action,) + state ] for action in self.actions }
        Q_action = max( actions_and_vals, key=actions_and_vals.get )

        return Q_action    ## Action with highest value in Q at the present state

    def semi_reckless(self, Q_action, state):
        wp = self.next_waypoint
        deadline = self.env.get_deadline(self)
        Q_value  = self.Q[ (Q_action,) + state ]   ## This is how Q rates the above Q_action (the actual max)
        wp_value = self.Q[ (wp,) + state ]         ## This is how Q rates the way_point

        # Making a time weighted choice given current state
        urgency = 1./(abs(deadline) + 1.)
        two_choices = {Q_action : Q_value * (1 - urgency), wp : wp_value * urgency }

        return max(two_choices, key=two_choices.get)

    #####################################################################################################
    #                       End Q-Learning
    #####################################################################################################

    def tally(self, reward, t):
        location = self.env.agent_states[self]['location']
        destination = self.env.agent_states[self]['destination']

        dist = self.env.compute_dist(location, destination)
        deadline = self.env.get_deadline(self)

        if reward < 0:                           ## Count bad moves
            self.bad_actions[self.trial] += 1

        if deadline < 1 or dist < 1:             ## Divide the number of bad moves by total moves in trial
            self.bad_actions[self.trial] /= 1.0*t
            self.total_time += t
           
            if deadline < 1 and dist >= 1:       ## Mark if agent ran out time of before reaching target
                self.out_of_times[self.trial] = 1
            else:
                self.out_of_times[self.trial] = 0

            if self.trial == number_trials - 1 : ## At the end of n trials;
                self.stats()                     ## plot bad_actions/actions ratio vs trial number
        
    def stats(self):

        out  = self.out_of_times                 ## 1 for out of time, 0 for reaching target
        bad_vals = self.bad_actions

        avg_trial = 1.0 * self.total_time/number_trials
        misses = sum(out), 100.* sum(out)/number_trials

        if self.no_plot:
            
            global cumulative_ts
            cumulative_ts[self.alpha, self.gamma] += avg_trial
            
        else:

            import matplotlib.pyplot as plt

            if sum(out) is not 0:
                # That is, if there are trials were agent missed the target
                x_miss, y_miss = zip(* [(i, x) for i,x in enumerate(bad_vals) if out[i] ] )
                plt.scatter(x_miss, y_miss, s=50, c="green", label="missed target")

                if sum(out) != len(out):
                    # If not ALL the trials were misses
                    x_hit, y_hit   = zip(* [(i, x) for i,x in enumerate(bad_vals) if not out[i] ] )
                    plt.scatter(x_hit , y_hit , s=50, c="red"  , label="reached target")
            else:

                plt.scatter(range(number_trials), bad_vals, s=50, c="red", label="reached target")

            ## Generate plot
            suptitle = "Policy: " + self.policy 
            title = "\navg trial length = {} \nmissed targets  = {} or {}%".format(avg_trial, misses[0], misses[1])
            plt.axis([0, number_trials, 0, 1])
            plt.suptitle(suptitle, fontweight='bold')
            plt.title(title)
            plt.xlabel("Learning Rate = {}".format(self.alpha))
            plt.legend(loc='right', bbox_to_anchor=(1, 1),prop={'size':10})
            plt.tight_layout()
            plt.show()


def run():
    """Run the agent for a finite number of trials."""
    runs = 1 #30
    for k in range(runs):
        for policy in ["random", "reckless", "Q_learning", "semi_reckless"]:
            # Set up environment and agent
            alpha, gamma = 1.0, 0.2     # After tinkering with many alpha/gamma pairs (see alternate main method below)
                                        # see pdf report for reasoning

            e = Environment()           # create environment (also adds some dummy traffic)
            a = e.create_agent(LearningAgent,policy,alpha, gamma, no_plot=False)  # create agent

            e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
            # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

            # Now simulate it
            sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
            # NOTE: To speed up simulation, reduce update_delay and/or set display=False

            sim.run(n_trials=number_trials)  # run for a specified number of trials
            # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

if __name__ == '__main__':
    run()

####################################################################################
    # Below, for testing values of learning rate alpha and discount rate gamma
    ##########################################################################

# from multiprocessing import Process, Manager
# manager = Manager()
# alphas = np.linspace(0.1, 1, 12)#[0.1, 0.5, 0.9]
# gammas = np.linspace(0.1, 0.99, 12)#[0.1, 0.5, 0.9]

# # Sums of avg times over n runs of 100 trials each
# cumulative_ts = manager.dict()

# def alpha_run(alp, gam):
#     policy = "Q_learning"
#     #policy = "semi_reckless"

#     # Set up environment and agent
#     e = Environment()
#     a = e.create_agent(LearningAgent, policy, alp, gam, no_plot=True)  # create agent
#     e.set_primary_agent(a, enforce_deadline=False)

#     # Now simulate it
#     sim = Simulator(e, update_delay=0.0, display=False)                # create simulator (uses pygame when display=True, if available)
#     sim.run(n_trials=number_trials)                                    # run for a specified number of trials
#     return


# if __name__ == '__main__':

#     runs = 70
#     jobs = []

#     for alp in alphas:                  #
#         for gam in gammas:              #
#             for k in range(runs):       # Faux Gridsearch
#                 cumulative_ts[alp,gam] = 0
#                 p = Process(target=alpha_run, args=(alp,gam))
#                 jobs.append(p)
#                 p.start()
#     for p in jobs:
#         p.join()

#     avg_2_target = { key : cumulative_ts[key]/runs for key in cumulative_ts.keys() }
#     minm = min(avg_2_target, key=avg_2_target.get)
#     minm = round(minm[0], 4), round(minm[1], 4)

#     ## Heat

#     import matplotlib.pyplot as plt

#     X, Y = np.meshgrid(alphas, gammas)
#     Z = np.array([ [np.log(avg_2_target[x,y]) for x in alphas] for y in gammas ])   ## At log scale;
#                                                                                     ##difference between values was too small, maps were washed out

#     print Z
#     print minm
#     plt.pcolor(X,Y,Z, cmap=plt.cm.Blues)
#     plt.axis([X.min(), X.max(), Y.min(), Y.max()])
#     plt.suptitle("Alpha-Gamma Heat Map", fontweight="bold")
#     plt.title("{} runs of {} trials each min at {}".format(runs, number_trials,minm) )
#     plt.xlabel("Alpha")
#     plt.ylabel("Gamma")
#     plt.colorbar()
#     plt.show()

