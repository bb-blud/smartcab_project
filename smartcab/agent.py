import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        self.lights  = ['green','red']
        self.gamma = 0.5
        self.Q = {

                  (action, light, waypoint, oncoming, left, right) : 0   \
                  for action   in self.actions                           \
                  for light    in self.lights                            \
                  for waypoint in self.actions                           \
                  for oncoming in self.actions                           \
                  for left     in self.actions                           \
                  for right    in self.actions                           \
        }
#        print "\n\n\n", self.Q[(None, 'green', 'left', None, None, None)], "\n\n"

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        wp = self.next_waypoint

        # # Update state
        state = (inputs['light'], wp, inputs['oncoming'], inputs['left'], inputs['right'])
        
        # # Select action according to policy
        current_actions = {action : self.Q[ (action,) + state ] for action in self.actions }
        Q_action = max( current_actions, key=current_actions.get ) # This is the action that has the highest value in Q at the present state 

        Q_value  = self.Q[ (Q_action,) + state ]   ## This is how Q rates the above Q_action (the actual max)
        wp_value = self.Q[ (wp,) + state ]         ## This is how Q rates the way_point
        
        # Making the best choice given our current state
        urgency = 1./(deadline + 0.001)
        two_choices = {Q_action : Q_value * (1- urgency), wp : wp_value * urgency }
        action = max(two_choices, key=two_choices.get)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.Q[ (action,) + state ] += reward * self.gamma

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
