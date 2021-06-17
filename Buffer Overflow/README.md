# Reinforcement Learning for Binary Exploitation, V0.02

Note: At the moment this is in development so documentation is sparse and packaging is not release standard! 

After you have installed the package(s) with:

> pip install -e bof_0_env

you can create an instance of the environment(s) with:

 - gym.make('bof_0:bof_0-v0'): 	Find a basic buffer overflow on a fixed-size buffer target.
 - gym.make('bof_0:bof_0-v01'): Find a basic buffer overflow on a variable-sze buffer target.

you can also execute train_bof0_model.py and run_bof0_model.py to train and test, respectively, an agent on the bof_0_env environemnt using A2C from stable_baselines3.