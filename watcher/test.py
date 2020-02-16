# import from_internet_or_for_from_internet.PNP_solver as pnp_solver
# import from_internet_or_for_from_internet.PNP_solver as pnp_solver

from predictor_module import GazeMLPredictor

if __name__ == "__main__":
    kek = GazeMLPredictor()
    while True:
        azaz = kek.predict_eye_vector_and_face_points(None, 0)
        if azaz[0][0] != 0:
            print(azaz)
