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
        self.Q_learner = {}
        self.reward = 0
        self.alpha = 0.6#0.2, 0.1,0.5,0.7
        self.gamma=0 ##no significant effects!!
        

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.reward = 0
	# TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.valid_actions = self.env.valid_actions

        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
                
        # TODO: Select action according to your policy
        
        available_actions = {action: self.Q_learner.get((self.state, action), 0) for action in self.valid_actions} # find all the actions and q-value in particular state
        actions = [action for action in self.valid_actions if available_actions[action] == max(available_actions.values())] ## choose action using max q-value
        #action = random.choice([None, 'forward', 'left', 'right'])
        #action = max(available_actions.iteritems(), key=lambda x:x[1])[0] ## find actions with max q-value
        action = random.choice(actions) #choose random action in can case same max q value is observed for multiple actions in the same iteration

        
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward +=reward
        # TODO: Learn policy based on state, action, reward
        # self.alpha * (reward+self.gamma * self.Q_learner.get((self.state_hat, action_hat), 0)) ## q - learning 
        self.Q_learner[(self.state, action)] =(1-self.alpha)*self.Q_learner.get((self.state, action), 0)+ self.alpha * reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
