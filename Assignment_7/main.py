import environment
import agent
import utilities

track = environment.Track("./L-track.txt", True)

player = agent.Q_Learn(track)

player.learn()
