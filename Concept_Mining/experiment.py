import random; random.seed(0)
import numpy as np; np.random.seed(0)

from utils import train_agent
# from Concept_Mining.ConceptMining import AdaptationEngine
import pickle,  pandas as pd
import seaborn as sns

from tqdm import tqdm
import optuna, os
import mlflow

def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)
    
def champion_callback(study, frozen_trial):
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            print(
                f"Trial {frozen_trial.number:2d} achieved value: {frozen_trial.value:.3f}"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
    

def run_training(settings: dict) -> tuple:

    average_reward, deviation_reward, history = train_agent(settings)
    return average_reward, deviation_reward, history

def optuna_reward( trial: optuna.Trial, settings: dict ) -> float:
    
    with mlflow.start_run(nested=True):

        hyperparameters = {
            "lr_critic" : trial.suggest_float('lr_critic', 2e-5, 1e-3),
            "temperature" : trial.suggest_float('temperature', 2, 5),
        }

        hyperparameters["lr_actor"] =  trial.suggest_float('lr_actor', 2e-5, 
                                                           hyperparameters["lr_critic"])
        hyperparameters["final_temperature"] = trial.suggest_float('final_temperature', 
                                                                   0.3, 0.5*hyperparameters["temperature"])


        mlflow.log_params(hyperparameters)
        average_reward, _, _ = run_training(settings | hyperparameters)

    mlflow.log_metric('average_reward', max(average_reward), step = trial.number)
    return max(average_reward)

BUFFER_SIZE = int(os.getenv('BUFFER_SIZE', None))


if __name__ == '__main__':

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    mlflow.set_tracking_uri(uri='http://localhost:8000')
    mlflow.set_experiment(f'Actor-Critic Buffer Size: {BUFFER_SIZE}')
    print(f'Experiment - Buffer size: {BUFFER_SIZE}')

    with mlflow.start_run(experiment_id=get_or_create_experiment(f'Actor-Critic Buffer Size: {BUFFER_SIZE}'),
                          run_name='optuna', nested=True):
        
        settings = {'episode_length': 1000,
            'episodes': 2000,
            'gamma': 0.99,
            'decay_rate': 0.99,
            'buffer_size': BUFFER_SIZE}
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: optuna_reward(trial, settings), n_trials=25)

        mlflow.log_params(study.best_params)
        mlflow.log_metric('average_reward', study.best_value)
        # _ = run_training( settings | st udy.best_params)

        # model = AdaptationEngine( settings | study.best_params )
        # model.Actor.load('actor.pt')
        # model.Critic.load('critic.pt')

        # # signature = mlflow.models.signature.infer_signature(df_train['text'].to_list(), 
        # #                                             model.predict(data = df_train['text'].to_list()))

        # model_info = mlflow.pytorch.log_model(model, artifact_path="ofenseval_learn", 
        #                                 signature=signature,
        #                                 registered_model_name="offenseval_learn_quickstart")
