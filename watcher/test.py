# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
import zerorpc
import cv2

class Roflan:
    def __init__(self):
        self.server = None
    def add_server(self, server):
        self.server = server
    def exit(self):

        self.server.stop()

if __name__ == "__main__":
    rofl = Roflan()
    zpc = zerorpc.Server(rofl)
    rofl.add_server(zpc)
    zpc.bind('tcp://127.0.0.1:8000')
    zpc.run()
    while True:
        pass
