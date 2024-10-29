# LRR (archived)

## Installation

The following are the necessary software dependencies for this project:

-   CUDA `=> 11.3`
-   Python `=> 3.10`
-   PyTorch `=> 2.0`

It is recommended to deploy the environment using both conda and docker. These installation instructions have been validated on docker images `nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04` and `nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04`.

Please use the following commands for installation:

```shell
conda create -n lrr python=3.10
conda activate lrr

conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt

# or
pip install cython opencv-python pyyaml h5py termtables tqdm typed_cap black scipy scikit-image Wand tensorboardX colorama visdom matplotlib torchattacks
pip install git+https://github.com/openai/CLIP.git

sudo apt update && sudo apt install -y libgl1-mesa-dev libmagickwand-dev
```

### Setting Environment Variables

Before initiating the training, testing, or evaluation processes, it's necessary to establish the correct environment variables. Here are five key variables to consider:

-   `PYOTP_PATH`
    This variable defines the root of the project/code submission and should not be changed.
-   `PYTHONPATH`
    By default, it's set to `$PYOTP_PATH/libs`, which informs the Python interpreter about the location of your local packages. This should not be changed.
-   `PYOTP_EXP`
    Default set to `$PYOTP_PATH/experiments`. It serves as a directory for running experiments and storing pretrained models. It can be modified according to user preference.
-   `TEST_DSET_PATH`
    Default set to `$PYOTP_PATH/data/test`. This path contains all testing datasets (including annotations) and can be changed if necessary.
-   `TRAIN_DSET_PATH`
    Default set to `$PYOTP_PATH/data/train`. This path houses all training datasets (including annotations) and can be altered if required.

You can use `source env.sh` to quickly set up the required environment variables.

### For Tracker

For tracker implementation, we utilize SiamRPN++ implementation from public GitHub repository [pysot](https://github.com/STVIR/pysot), which also offers setup instructions for training and testing datasets.

To set up the tracker, you need to build the custom [extensions](https://github.com/STVIR/pysot/blob/master/INSTALL.md#build-extensions) provided by pysot. We have included the pre-built extension for Python 3.10. If there are any issues, you can rebuild it locally with the following commands:

```shell
cd libs

# remove the current built
rm toolkit/utils/region.c
rm toolkit/utils/region.cpython-310-x86_64-linux-gnu.so

# build the extension
python toolkit_setup.py build_ext --inplace
```

Additionally, you must download the pretrained tracker models from pysot's [model zoo](https://github.com/STVIR/pysot/blob/master/MODEL_ZOO.md). Specifically, you need the models

-   `siamrpn_r50_l234_dwxcorr`
-   `siamrpn_r50_l234_dwxcorr_otb`
-   `siamrpn_mobilev2_l234_dwxcorr`

After downloading, these models should be placed into the corresponding directories under `$PYOTP_EXP/pysot/pretrained`.

### For Attacker

The attacker, CSA, requires additional setup. First, you need to install the dependency [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) for CSA.

We provide a helper script that simplifies the installation of pix2pix:

```shell
# run this in project's root directory
bash $PYOTP_PATH/scripts/install/csa_libs.sh
```

Secondly, you need to download the pretrained CSA generators following the official [instruction](https://github.com/MasterBin-IIAU/CSA#download-pretrained-models). After downloading the pretrained generators, they should be placed into `$PYOTP_EXP/CSA/checkpoints`.

## Testing

Before commencing the testing phase, ensure you have downloaded and properly set up the testing datasets. For further details, you can consult pysot's [documentation](https://github.com/StrangerZhang/pysot-toolkit#download-dataset).

<details>
<summary>More about testing dataset download</summary>

| Dataset      | URL                                                               | Desc                                                                                                                                           |
| ------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| VOT2019      | https://drive.google.com/file/d/1wsLBHP6ssu5CffnnRFf4NTYatWQqrj8N |                                                                                                                                                |
| OTB100       | http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html        | Alternative: using script from [MMTracking](https://github.com/open-mmlab/mmtracking/blob/master/docs/en/dataset.md#13-single-object-tracking) |
| LaSOT        | https://vision.cs.stonybrook.edu/~lasot/download.html             | Download the testsets only                                                                                                                     |
| UAV123       | https://cemse.kaust.edu.sa/ivul/uav123                            |                                                                                                                                                |
| NFS30/NFS240 | `curl -fSsl http://ci2cv.net/nfs/Get_NFS.sh           \| bash -`  | Official [script](http://ci2cv.net/nfs/index.html)                                                                                             |

</details>

Once your testing datasets are prepared, you can initiate test tracking under the respective attacker-specific directories. Use the following directories for each type of attack:

-   Without attack: `$PYOTP_EXP/pysot`
-   CSA: `$PYOTP_EXP/CSA`
-   IoU Attack: `$PYOTP_EXP/IoUAtk`
-   RTAA: `$PYOTP_EXP/RTAA`
-   SPARK: `$PYOTP_EXP/SPARK`

### Non-Defense Testing

To execute non-defense tracking, employ the following command:

```shell
python $PYOTP_PATH/scripts/tests/no_defense.py \
    -t r50 -A csa -d NFS30
```

Here, the parameters signify the following:

-   `-t/--tracker`: Valid values include `r50` and `mob`, representing SiamRPN++ with a ResNet50 backbone or MobileNetV2 backbone, respectively. This will save raw tracking results in `./results_r50` and `./results_mob`.
-   `-A/--attacker`: Accepts `none`, `cas`, `iou`, `rtaa`, and `spark,TA`, defining the class of attacker.
-   `-d/--datasets`: Valid dataset options include `VOT2019`, `OTB100`, `UAV123`, `NFS30`, `NFS240`, and `LaSOT`.
-   `-V/--visualization` (_Optional_): This flag generates visualizations in the raw results directory, storing tracking images in `${raw_results}/vis_trkpp`.
-   `-s/--suffix` (_Optional_): Adds a suffix to result directory names.
-   `-f/--force` (_Optional_): This flag allows overwriting existing results with the same name.

### Defense Testing

To run defense tracking with pretrained LRR (`$PYOTP_EXP/lrr_saves/lrr_pretrained/lrr-epoch-best.pth`), use the following command:

```shell
python $PYOTP_PATH/scripts/tests/defense_with_LRR.py \
    -t r50 -A csa -d NFS30
```

This script accepts the same command-line arguments as the previous `non-defense.py`.

## Evaluation

To assess tracking performance on a specific testing dataset, run the following command:

```shell
python $PYOTP_PATH/libs/pyotp/tools/eval.py \
    -p results_r50 \
    -d NFS30 \
    -t '*' -v '*'
```

Here, the parameters signify the following:

-   `-p/--tracker-path`: Specifies the path of your results directory, corresponding to the previous testing results directories `./results_r50` and `./results_mob`.
-   `-d/--dataset`: Valid dataset options include `VOT2019`, `OTB100`, `UAV123`, `NFS30`, `NFS240`, and `LaSOT`.
-   `-t/--tracker-filter`: This option filters tracking results with a wildcard. Use `'*'` (single quotes to prevent shell completion) to match all results.
-   `-v/--variant-filter`: This option filters tracking results on variants with a wildcard. Use `'*'` (single quotes to prevent shell completion) to match all variants.

## Training

### Preparation

To reproduce the training of STIR and LResampleNet, you need to generate training data pairs with FGSM. If you also aim to reproduce the refine (adversarial training) baseline, you will require additional training data pairs perturbed by PGD and CSA. Start by downloading and cropping the clean training data - refer to the official pysot's [documentation](https://github.com/STVIR/pysot/blob/master/TRAIN.md) for guidance. Afterwards, use our script to generate the paired training data. We'll use the path `$PYOTP_PATH/data/train_mod/$ATK` for all following examples, but you can replace it with any location of your preference. The data root for the training process can be specified in the training config files.

To generate paired training data with perturbation, set the correct command-line arguments and generation config. You can select a dataset for use by changing the config file `configs/siamrpnpp/train_data_explore.py`, leaving the dataset you want to generate in the `cfg.dataset.names` list and commenting out the others.

-   For single frame datasets like DET and COCO, use `--n_length 1` (or `-N 1`). Extra frames used in training STIR or LResampleNet will be synthesized by the dataloader. Use this command:

    ```shell
    ATK=FGSM
    N=1
    REPEAT=1

    python $PYOTP_PATH/libs/pyotp/datasets/tools/create_adv_dataset.py \
        -c configs/siamrpnpp/train_data_explore.py \
        -s $PYOTP_EXP/pysot/pretrained/siamrpn_r50_l234_dwxcorr/model.pth \
        -N $N -r $REPEAT \
        -A $ATK \
        -O $PYOTP_PATH/data/train_mod/$ATK/output.h5
    ```

-   For sequence frames datasets like VID and YOUTUBEBB, use `--n_length 5` (or `-N 5`) to generate a sequence of paired data. Since frames are randomly selected from a sequence, you'll need to iterate over the whole dataset multiple times to achieve the desired training dataset size. In the following example, we set `REPEAT=10` to iterate over the dataset 10 times.

    ```shell
    ATK=FGSM
    N=5
    REPEAT=10

    python $PYOTP_PATH/libs/pyotp/datasets/tools/create_adv_dataset.py \
        -c configs/siamrpnpp/train_data_explore.py \
        -s $PYOTP_EXP/pysot/pretrained/siamrpn_r50_l234_dwxcorr/model.pth \
        -N $N -r $REPEAT \
        -A $ATK \
        -O $PYOTP_PATH/data/train_mod/$ATK/output.h5
    ```

Generation of paired training data with perturbations can be time-consuming, so consider setting a range of dataset indexing to distribute your generation process across multiple machines.

Once you've generated the paired training data, create an index in JSON format, which is required by the training dataloader. Use the following command:

```shell
python $PYOTP_PATH/libs/pyotp/datasets/tools/gen_h5_index.py \
    -d VID,YOUTUBEBB \
    -o $PYOTP_PATH/data/train_mod/$ATK/index_vid_bb.json \
    $PYOTP_PATH/data/train_mod/$ATK/*.h5
```

This step will potentially generate duplicate pairs. The index also removes duplicates across all input hdf5 files.

### Training of STIR

Once the paired training data is prepared

, you can train STIR on your host. You'll first need to define the data files and indexes you want to use in the training config file. You can modify `$PYOTP_PATH/configs/LRR/train/stir.yml` to do this.

You'll need to set the correct path of your data files in the config since training data are generated locally and with your own splits. In the config file, first set `dataset.args.root` for both `train_dataset` and `val_dataset`. Then list all data files you want to use in `dataset.args.h5fp` with their relative path to the data root. Finally, list corresponding index files in `dataset.args.cache_index_fp`.

To train the STIR network, use the following command:

```shell
python $PYOTP_PATH/libs/LRR/tools/train.py \
    --config $PYOTP_PATH/configs/LRR/train/stir.yml \
    --name LRR/STIR \
    --trial 0
```

### Training of LRR

After training the STIR network, you can train the LResampleNet over the trained STIR to get the LRR network. Also change the corresponding data file settings in the config file `$PYOTP_PATH/configs/LRR/train/lrr.yml`. LRR shares the same items in STIR's config, with the addition of setting `dataset.args.object_cls_files`, which contains the language guidance label of the template for the LResampleNet training in JSON. The format of this JSON file is `dict[str, str]`.

To train the LRR network, use the following command:

```shell
python $PYOTP_PATH/libs/LRR/tools/train_rsn.py \
    -c $PYOTP_PATH/configs/LRR/train/lrr.yml \
    -s $STIR_saves/epoch-best.pth \
    -n LRR/LResampleNet \
    --trial 0 \
    --rsn-type cnn \
    --layers 256,32
```

### Adversarial Training Tracker Network

In order to refine the tracker network with pre-generated paired data, you need 1) paired data stored in hdf5 files and 2) indexes of the paired data in JSON format. You can find a usage example of how to import your data files into the training process in line 107 at `$PYOTP_PATH/libs/pyotp/tools/train.py`.

For adversarial training on a specific tracker, you should use the corresponding config file. You can find the config files in `$PYOTP_PATH/configs/siamrpnpp/*_refine_iclr.py`. For SiamRPN++ with ResNet50 backbone tracker, note that there is an OTB variant specifically designed for evaluation on the OTB100 dataset.

You can initiate the adversarial training process with the following command:

```shell
# trained models can be found in $PYOTP_EXP/refine_iclr
torchrun --nproc_per_node=1 --master_port=2133 \
    $PYOTP_PATH/libs/pyotp/tools/train.py \
    -c $PYOTP_PATH/configs/siamrpnpp/train_trktp_refine_iclr.py \
    -b 32
```

## Bibtex

```
@article{chen2024lrr,
  title={LRR: Language-Driven Resamplable Continuous Representation against Adversarial Tracking Attacks},
  author={Chen, Jianlang and Ren, Xuhong and Guo, Qing and Juefei-Xu, Felix and Lin, Di and Feng, Wei and Ma, Lei and Zhao, Jianjun},
  journal={ICLR},
  year={2024}
}
```
