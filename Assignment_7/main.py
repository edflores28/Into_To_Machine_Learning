import environment
import agent
import argparse


MAX_EPISODES = 100


def do_l_track():
    print("Running the L Track")
    # Create the track and set the crash policy
    # to start at the nearest crash position
    l_track = environment.Track(filename="./L-track.txt", is_start=False, is_ltrack=True)
    # Create the agent for Q Learning
    p_Q = agent.MDP(track=l_track, episodes=MAX_EPISODES, is_qlearn=True)
    print("Using Q learning")
    input("Press Enter to continue")
    # Train the agent
    p_Q.learn()
    print("Testing Q Learning")
    # Testing the model
    p_Q.test()
    # Create the agent for SARSA
    p_S = agent.MDP(track=l_track, episodes=MAX_EPISODES, is_qlearn=False)
    print("Using SARSA")
    input("Press Enter to continue")
    p_S.learn()
    print("Testing SARSA")
    p_S.test()


def do_r_track():
    print("Running the R Track")
    # Create the track and set the crash policy
    # to start at the nearest crash position
    r_track = environment.Track(filename="./R-track.txt", is_start=False, is_ltrack=False)
    # Create the agent for Q Learning
    p_Q = agent.MDP(track=r_track, episodes=MAX_EPISODES, is_qlearn=True)
    print("Using Q learning with start at finish line false")
    input("Press Enter to continue")
    # Train the agent
    p_Q.learn()
    print("Testing Q Learning")
    # Testing the model
    p_Q.test()
    # Create the agent for SARSA
    p_S = agent.MDP(track=r_track, episodes=MAX_EPISODES, is_qlearn=False)
    print("Using SARSA start at finish line false")
    input("Press Enter to continue")
    p_S.learn()
    print("Testing SARSA")
    p_S.test()
    # Create the track and set the crash policy
    # to start at the nearest crash position
    r_track = environment.Track(filename="./R-track.txt", is_start=True, is_ltrack=False)
    # Create the agent for Q Learning
    p_Q = agent.MDP(track=r_track, episodes=MAX_EPISODES, is_qlearn=True)
    print("Switching the crashing policy")
    print("Using Q learning with start at finish line True")
    input("Press Enter to continue")
    # Train the agent
    p_Q.learn()
    print("Testing Q Learning")
    # Testing the model
    p_Q.test()
    # Create the agent for SARSA
    p_S = agent.MDP(track=r_track, episodes=MAX_EPISODES, is_qlearn=False)
    print("Using SARSA start at finish line True")
    input("Press Enter to continue")
    p_S.learn()
    print("Testing SARSA")
    p_S.test()


'''
MAIN APPLICATION
'''
# Create a parser for the command line arguments
parser = argparse.ArgumentParser(description="Intro to ML Project 7")
parser.add_argument('-l', action="store_true", default=False, help='Execute L racetrack')
parser.add_argument('-r', action="store_true", default=False, help='Execute R racetrack')

results = parser.parse_args()

# Perform the tests based on the input
if results.l:
    do_l_track()
if results.r:
    do_r_track()
