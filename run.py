import sys
import os
sys.path.append("./experiments")
import mac.utils
from mac.mac import mac
import mac.run as run
import experiments.run_learning_experiment as run_learning_experiment
from icml_2019_state_abstraction.experiments import run_learning_experiment


def main():

    gym_env = sys.argv[1]

    ## run training of policy
    ## run.main()

    ## run learning experiment
    run_learning_experiment.main(gym_env, abstraction=True)

if __name__ == "__main__":
    main()