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

# Download files

## Download model files

- Download model files from [here](http://www.di.ens.fr/willow/research/obman/release_models.zip) `wget http://www.di.ens.fr/willow/research/obman/release_models.zip`
- unzip `unzip release_models.zip`

## Download the MANO model files

- Go to [MANO website](http://mano.is.tue.mpg.de/)
- Create an account by clicking *Sign Up* and provide your information
- Download Models and Code (the downloaded file should have the format mano_v*_*.zip). Note that all code and data from this download falls under the [MANO license](http://mano.is.tue.mpg.de/license).
- unzip and copy the content of the *models* folder into the misc/mano folder


- Your structure should look like this:

```
image2mano_params/
  misc/
    mano/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
  release_models/
    fhb/
    obman/
    hands_only/
```


# Launch

python get_th_full_pose.py --image_path image_path --hand_side hand_side (--flip)


# Acknowledgements

The code is adopted from [obman_train](https://github.com/hassony2/obman_train) with only a few modifications.