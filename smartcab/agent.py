import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator



number_trials = 100

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, policy, alpha, gamma, no_plot):
        super(LearningAgent, self).__init__(env)     # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'                           # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.policy = policy

        # #Initialize any additional variables here

        # State descriptors
        self.actions = self.env.valid_actions
        self.lights  = ['green','red']

        # For tallying performance and making figures for the report
        self.bad_actions  = {}
        self.out_of_times = {}
        self.trial = -1 
        self.total_time = 0
        self.no_plot = no_plot   # activate plots or not
        
        # For Q learning implementation
        self.gamma = gamma
        self.alpha = alpha
        
        self.Q = {

                  (action, light, waypoint, oncoming, left, right) : 0   \
                  for action   in self.actions                           \
                  for light    in self.lights                            \
                  for oncoming in self.actions                           \
                  for left     in self.actions                           \
                  for right    in self.actions                           \
                  for waypoint in self.actions[1:]                       # waypoint is only None when target is reached
        }

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()           ## from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        wp = self.next_waypoint
        
        # # Update state
        state = (inputs['light'], wp, inputs['oncoming'], inputs['left'], inputs['right'])
        
        # # Select action according to policy
        Q_action = self.max_action(state)

        action  = {
                   "random"        : random.choice(self.actions[1:]),
                   "reckless"      : wp,
                   "semi_reckless" : self.semi_reckless(Q_action, state),
                   "Q_learning"    : Q_action } [self.policy]      ## Dictionary of different policies for comparison

        # # Execute action and get reward
        reward = self.env.act(self, action)

        # # Learn policy based on state, action, reward
        alpha = self.alpha      ## Learning rate

        new_state = (inputs['light'], self.next_waypoint, inputs['oncoming'], inputs['left'], inputs['right'])
        new_action = self.max_action(new_state)

        V = self.Q[ (action,) + state ]
        X = reward + self.gamma * self.Q[ (new_action,) + new_state]

        self.Q[ (action,) + state ] =  (1-alpha) * V + alpha *X   ## Updating Q

        # # Tally bad actions
        self.tally(reward, t)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
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
        urgency = 1./(deadline + 0.001)
        two_choices = {Q_action : Q_value * (1 - urgency), wp : wp_value * urgency }

        return max(two_choices, key=two_choices.get)


    def tally(self, reward, t):
        location = self.env.agent_states[self]['location']
        destination = self.env.agent_states[self]['destination']

        dist = self.env.compute_dist(location, destination)
        deadline = self.env.get_deadline(self)
        
        if t == 0:                           ## Start tally for new trial
            self.trial += 1
            self.bad_actions[self.trial] = 0

        if reward < 0:                       ## Count bad moves
            self.bad_actions[self.trial] += 1
            
        if deadline < 1 or dist < 1:         ## Divide the number of bad moves by total moves in trial
            self.bad_actions[self.trial] /= 1.0*t
            self.total_time += t
           
            if deadline < 1 and dist >= 1:   ## Mark if agent ran out time of before reaching target
                self.out_of_times[self.trial] = 1
            else:
                self.out_of_times[self.trial] = 0 

                if self.trial == number_trials - 1 :
                    self.stats()             ## Plot bad_actions/actions ratio vs trial number
        
    def stats(self):

        out  = self.out_of_times.values()          ## 1 for out of time, 0 for reaching target
        vals = self.bad_actions.values()

        avg_trial = 1.0 * self.total_time/number_trials
        misses = sum(out), 100.* sum(out)/number_trials

        if self.no_plot:
            
            global cumulative_ts
            cumulative_ts[self.alpha, self.gamma] += avg_trial

            # if (self.alpha, self.gamma) not in cumulative_ts.keys():
            #     cumulative_ts[self.alpha, self.gamma] = []
            # else:
            #     cumulative_ts[self.alpha, self.gamma].append(avg_trial)
            
        else:

            import matplotlib.pyplot as plt

            if sum(out) is not 0:   # That is, if there are trials were agent missed the target

                x_miss, y_miss = zip(* [(i, x) for i,x in enumerate(vals) if out[i] ] )
                x_hit, y_hit   = zip(* [(i, x) for i,x in enumerate(vals) if not out[i] ] )

                plt.scatter(x_hit , y_hit , s=50, c="red"  , label="reached target")
                plt.scatter(x_miss, y_miss, s=50, c="green", label="missed target")

            else:

                plt.scatter(range(number_trials), vals, s=50, c="red", label="reached target")

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

    for policy in ["random", "reckless", "semi_reckless", "Q_learning"]:
        # Set up environment and agent
        alpha, gamma = 1.0, 0.6     # After inkering with many alpha/gamma pairs (see alternate main method below)
                                    # gamma is average of 4 and 8 (see pdf report)

        e = Environment()           # create environment (also adds some dummy traffic)
        a = e.create_agent(LearningAgent,policy,alpha, gamma, no_plot=False)  # create agent

        e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
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
# gammas = np.linspace(0.1, 0.9, 12)#[0.1, 0.5, 0.9]

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
#     sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
#     sim.run(n_trials=number_trials)  # run for a specified number of trials
#     return


# if __name__ == '__main__':

#     runs = 15
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

        
#     avg_2_target = { key : round(cumulative_ts[key]/runs, 2)  for key in cumulative_ts.keys() }
#     minm = min(avg_2_target, key=avg_2_target.get)

#     ## Heat

#     import matplotlib.pyplot as plt

#     X, Y = np.meshgrid(alphas, gammas)
#     Z = np.array([ [avg_2_target[x,y] for x in alphas] for y in gammas ])
#     print Z
#     print minm
#     plt.pcolor(X,Y,Z, cmap=plt.cm.Blues)
#     plt.axis([X.min(), X.max(), Y.min(), Y.max()])
#     plt.suptitle("Alpha-Gamma Heat Map", fontweight="bold")
#     plt.title("Minimum at {}".format(minm))
#     plt.xlabel("Alpha")
#     plt.ylabel("Gamma")
#     plt.colorbar()
#     plt.show()

