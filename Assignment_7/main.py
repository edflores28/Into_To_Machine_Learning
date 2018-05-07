import environment
import agent
import utilities

track = environment.Track("./R-track.txt")

player = agent.Q_Learn(track)

player.learn()
