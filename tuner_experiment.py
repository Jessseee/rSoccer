from dataclasses import dataclass

import optuna
import tyro

from cleanrl_utils.tuner import Tuner


@dataclass
class Args:
    env_id: str = None
    """the id of the environment"""
    model: str = None
    """Path to the CleanRL model python file"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    wandb_mode: str = "online"
    """The wandb mode"""
    wandb_dir: str = "./"
    """The wandb output directory"""
    run_dir: str = "./runs"
    """The summary output directory"""


if __name__ == "__main__":
    args = tyro.cli(Args)

    tuner = Tuner(
        script=args.model,
        metric="charts/episodic_return",
        metric_last_n_average_window=50,
        direction="maximize",
        aggregation_type="average",
        target_scores={
            args.env_id: None,
        },
        params_fn=lambda trial: {
            "learning-rate": trial.suggest_float("learning-rate", 3e-4, 3e-3, log=True),
            "policy_frequency": trial.suggest_categorical("policy_frequency", [1, 2, 4, 8]),
            "exploration_noise": trial.suggest_float("exploration_noise", 0.1, 0.2),
            "total-timesteps": 500000,
            "capture_video": args.capture_video,
            "track": True,
            "wandb_mode": args.wandb_mode,
            "wandb_project_name": args.wandb_project_name,
            "wandb_dir": args.wandb_dir,
            "wandb_entity": args.wandb_entity,
            "run_dir": args.run_dir,
        },
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(),
        wandb_kwargs={
            "project": args.wandb_project_name,
            "mode": args.wandb_mode,
            "dir": args.wandb_dir,
            "entity": args.wandb_entity,
        },
    )
    tuner.tune(
        num_trials=25,
        num_seeds=3,
    )
