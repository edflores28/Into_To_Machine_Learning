import environment
import agent

track = environment.Track("./L-track.txt", True)

player = agent.Q_Learn(track)

player.learn()
