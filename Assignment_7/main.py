import environment
import agent
import utilities

track = environment.Track("./L-track.txt")

player = agent.Q_Learn(track)

player.learn()
