import wandb

wandb.login()

run = wandb.init(
    entity="fencepainters",
    project="evraz",
    job_type="data-logging",
    notes="log dataset"
)

my_data = wandb.Artifact("RawDataset", type="dataset", description="Just raw files")

my_data.add_dir("data")
run.log_artifact(my_data)

wandb.finish()