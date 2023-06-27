import torch as t
from data import GestureDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import torch_model as models
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import lr_scheduler
import mlflow
from focal_loss.focal_loss import FocalLoss, FocalLoss2

experiments = [
    {
        "include_context": True,
        "include_raw_kpts": True,
        "include_skltn": True,
        "include_linear_distances": True,
        "name": "context_kpt_skltn_distances",
    },
    {
        "include_context":True,
        "include_raw_kpts": True,
        "include_skltn": True,
        "include_linear_distances": False,
        "name": "context_kpt_skltn",
    },
    {
        "include_context":True,
        "include_raw_kpts": False,
        "include_skltn": True,
        "include_linear_distances": True,
        "name": "context_skltn_distances",
    },
    {
        "include_context":False,
        "include_raw_kpts": True,
        "include_skltn": True,
        "include_linear_distances": True,
        "name": "kpts_skltn_distances",
    },
    
    {
        "include_context":True,
        "include_raw_kpts": True,
        "include_skltn": False,
        "include_linear_distances": False,
        "name": "context_kpts",
    },
    {
        "include_context":True,
        "include_raw_kpts": False,
        "include_skltn": False,
        "include_linear_distances": True,
        "name": "context_distances",
    },
    {
        "include_context":True,
        "include_raw_kpts": True,
        "include_skltn": False,
        "include_linear_distances": False,
        "name": "context_kpts",
    },
    {
        "include_context": False,
        "include_raw_kpts": False,
        "include_skltn": False,
        "include_linear_distances": False,
        "name": "crop_only",
    },
    # {
    #     "include_context":True,
    #     "include_raw_kpts": False,
    #     "include_skltn": False,
    #     "include_linear_distances": False,
    #     "name": "img_skltn",
    # },
    # {
    #     "include_context":True,
    #     "include_raw_kpts": False,
    #     "include_skltn": False,
    #     "include_linear_distances": False,
    #     "name": "img_dist",
    # },
]
exp_name = "gesture_classification_context"
# mlflow.set_tracking_uri("http://localhost:5000/")

for exp in experiments:
    print(exp, '\n')

    experiment = mlflow.set_experiment(experiment_name=exp_name)
    run = mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=exp["name"])
    run = mlflow.active_run()
    run_id = run.info.run_id

    # load the data from the csv file and perform a train-test-split
    # this can be accomplished using the already imported pandas and sklearn.model_selection modules
    # "/net/cluster/azhar/mywork/datasets/task_smellpersons_personbboxes/images/"
    images_path_train = "./data/data_context_train_filtered/"
    images_path_val = "./data/data_context_test_filtered/"
    context_images_path = "/net/cluster/azhar/mywork/datasets/task_smellpersons_personbboxes/all_images"
    # data_path = "/net/cluster/azhar/Keypoints-Classification/data/postprocessed_df_N-H_min_area_filtered_area_filter_merged.csv"
    # data = pd.read_csv(data_path)
    # data = data[data["label"]!=2] # excluding sniffing calss
    # train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects

    # hyperparameters
    input_size = (224, 224)  # (224, 224)
    input_channels = 3
    batch_size = 28
    num_classes = 1  # len(np.unique(data["label"].values))
    epochs = 100
    early_stopping_patience = 5
    num_workers = 4
    shuffle = True
    checkpoint_path = "checkpoints/{}_resnet18_skelton_cleaned_data.ckp".format(
        run_id)
    lr = 0.001
    mom = 0.9
    include_context = exp["include_context"]
    include_raw_kpts = exp["include_raw_kpts"]
    include_skltn = exp["include_skltn"]
    include_linear_distances = exp["include_linear_distances"]
    if num_classes ==2:
        class_names = ["background", "holding_nose"]
    else:
        class_names = ["holding nose"]

    mlflow.log_param("run_id", run_id)
    mlflow.log_param("input_size", input_size)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("include_skelton", include_skltn)
    mlflow.log_param("include_linear_distances", include_linear_distances)
    mlflow.log_param("include_context", include_context)
    mlflow.log_param("include_raw_keypoints", include_raw_kpts)
    mlflow.log_param("train_path", images_path_train)
    mlflow.log_param("class_names", class_names)

    train_dataset = GestureDataset(
        images_path_train,
        context_images_path,
        input_size,
        num_classes,
        mode="train",
        include_raw_keypoints=include_raw_kpts,
        include_distances=include_linear_distances,
        include_context=include_context,
        include_skelton=include_skltn,
    )
    train_dataloader = t.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    val_dataset = GestureDataset(
        images_path_val,
        context_images_path,
        input_size,
        num_classes,
        mode="val",
        include_raw_keypoints=include_raw_kpts,
        include_distances=include_linear_distances,
        include_context=include_context,
        include_skelton=include_skltn,
    )
    val_dataloader = t.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    in_channels = train_dataset.in_channels
    # create an instance of our ResNet model
    class_weights = t.tensor(train_dataset.class_weights)
    model = models.CustomModel(
        num_classes,
        input_channels=input_channels,
        include_raw_keypoints=include_raw_kpts,
        include_distances=include_linear_distances,
        include_context=include_context,
        include_skelton=include_skltn,
    )
    # set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
    if num_classes > 1:
        criterion = t.nn.CrossEntropyLoss(
            weight=class_weights
        )  
    else:
        criterion = t.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    # criterion = FocalLoss2()
    # criterion = FocalLoss(gamma=0.7, weights=class_weights.cuda())
    # criterion = torch.hub.load(
    #     'adeelh/pytorch-multi-class-focal-loss',
    #     model='FocalLoss',
    #     alpha=class_weights,
    #     gamma=2,
    #     reduction='mean',
    #     force_reload=False
    # )
    # set up the optimizer (see t.optim)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)#, momentum=mom)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # set logging here

    mlflow.log_param("loss", str(type(criterion)))
    mlflow.log_param("optimizer", str(type(optimizer)))
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("in_channels", in_channels)

    # create an object of type Trainer and set its early stopping criterion
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        train_dataloader,
        val_dataloader,
        cuda=True,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        class_names=class_names,
    )

    # # go, go, go... call fit on trainer
    res = trainer.fit(epochs)

    mlflow.end_run()
