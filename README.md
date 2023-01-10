# Install

```
git clone https://github.com/freshman0213/image2mano_params.git
cd images2mano_params
git submodule update --init --recursive
```


# Setup python environment

- create conda environment with dependencies: `conda env create -f environment.yml`
- activate environment: `conda activate image2mano_params`
- install manopth:
  ```
  cd modified_manopth
  pip install .
  ```
- install mano_pybullet:
  ```
  cd mano_pybullet
  pip install -e .
  ```

# Download files

## Download model files

- Download model files from [here](http://www.di.ens.fr/willow/research/obman/release_models.zip) `wget http://www.di.ens.fr/willow/research/obman/release_models.zip`
- unzip `unzip release_models.zip`

## Download the MANO model files

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the content of the `models` folder into the `misc/mano` folder

## Download the DexYCB dataset

- Download the DexYCB dataset [here](https://drive.google.com/file/d/1YhbSyuWB4JpANorp2E6hwzaqqU7drEfy/view?usp=sharing)
- Place the file under `misc/`
- Extract the .zip file

## Download hand images

- Download the example hand images [here](https://drive.google.com/file/d/1uZQrjsguuNuaNLiqvvN8NOk-6iH_9O8b/view?usp=sharing)
- Place the file under `misc/`
- Extract the .zip file

## Final file structure
```
image2mano_params/
  release_models/
    fhb/
    obman/
    hands_only/
  misc/
    mano/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
    dex-ycb/
    hand_images/
```

# Launch

```
python get_th_full_pose.py --image_path image_path --hand_side hand_side (--flip)
python get_pose_m.py --th_full_pose_path th_full_pose_path --hand_side hand_side
```

# Acknowledgements

The code is adopted from [obman_train](https://github.com/hassony2/obman_train) with only a few modifications.
