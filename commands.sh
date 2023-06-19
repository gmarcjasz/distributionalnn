# commands tested on GNU/Linux systenm with Python 3.9 installed
# commands should be run from within `Python` folder

#####
# For the LEAR-QRA model, takes ca. 10 hours on 16-core workstation
python3 lear.py DE 56
python3 lear.py DE 84
python3 lear.py DE 1092
python3 lear.py DE 1456
python3 eval_lear.py
python3 QRA.py

#####
# For the DDNN-JSU model
# rolling calibration based on the attached hyper-parameter optimization trialfiles
python3 nn_rolling_pred.py DE JSU
# the code above will run the trial `FINAL_DE_selection_prob_jsu` - trials 1..4 are attached
# one has to be copied to the filename without trial number suffix
# NN codes take hours to run the rolling prediction (hyperparameter optimization takes weeks/months)

# NN results are not replicating the values (the random seeds were not set due to parallel execution)
# but should yield qualitatively similar results upon consecutive runs
