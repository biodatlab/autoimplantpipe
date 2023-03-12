## Training scripts

`scripts` folder includes the script for training models and baseline models.

### Autosegmentation

- `train_autosegmentation_pl.ipynb` is a notebook for training a segmentation model with Pytorch Lightning

You can run all cells in the notebook to fine-tune the model.

### Autopimplant

- `train_autoimplant.py` is a script for training an autoimplant model

You can run the following commands to fine-tune the model.
**Note** that you may need to login to [wandb](https://wandb.ai/) before running the script.

```sh
python train_autoimplant.py --df_data <data_path_csv> --pretrain_path <pretrain_path> --batch_size <batch_size> --n_epochs <n_epochs>
```

**Arguments:**

- `<data_path_csv>` is a path to CSV file which contained `defect_skull_path` and `complete_skull_path` columns
- `<pretrain_path>` is a pretrained path to autoimplant model
- `<batch_size>` is a batch size for the training
- `<n_epochs>` is the number of training epochs
- `--no-cuda` flag to disable CUDA training
