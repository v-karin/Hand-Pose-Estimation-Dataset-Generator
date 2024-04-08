# Hand Pose Estimation: Dataset Generator

## Requirements

Steps to download MANO model:
- Go to [MANO Website](http://mano.is.tue.mpg.de/) and create an account.
- Download Models and Code (the downloaded file should be named `mano_v1_2.zip`).
- Make a `mano` folder in the `Dataset Generator` folder.
- Unzip and copy the contents in `mano_v1_2/` folder to the `Dataset Generator/mano/` folder.
- Then, copy the `smpl_handpca_wrapper_HAND_only.py` and `verts.py` files from the [manotorch Repository](https://github.com/lixiny/manotorch/tree/master/mano/webuser) and paste them in the `Dataset Generator/mano/webuser` folder, replacing any previous files.
- Your folder structure should look like this:

```
Dataset Generator/
  mano/
    models/
      MANO_LEFT.pkl
      MANO_RIGHT.pkl
  webuser/
    smpl_handpca_wrapper_HAND_only.py
    serialization.py
    verts.py
    ...
```

Porting from Python 2.7 to Python 3 (`serialization.py`):
- in `serialization.py`, replace `import cPickle as pickle` with `import pickle`
- Then, replace `pickle.dump(trainer_dict, open(fname, 'w'), -1)` with `pickle.dump(trainer_dict, open(fname, 'wb'), -1)`

## Installation

Steps to get the Dataset Generator to work:
- pip install blenderproc
- then do `blenderproc quickstart` in the Command Prompt
- this will automatically install Blender 3.5


- open dataset_canvas.blend with Blender 3.5
- go to script section
- run the script

Last step:
- pip install chumpy 0.71 (use git repo)

`pip install git+https://github.com/mattloper/chumpy.git`

- go to your site-packages directory for your default python installation
e.g. C:\Users\\[user]\AppData\Local\Programs\Python\Python310\Lib\site-packages
- copy the "chumpy" folders i.e. `chumpy`, `chumpy-0.71.dist-info`
- paste them in your site-packages directory for your Blender python installation
e.g. C:\Users\\[user]\blender\blender-3.5.1-windows-x64\blender-3.5.1-windows-x64\3.5\python\lib\site-packages

this will install Chumpy 0.71, default pypi website only provides Chumpy 0.70, which has an obsolete incompatibility bug with new versions of numpy






Initial Cwd (Command Prompt Working Directory):
`C:/.../[insert user]/.../Hand-Pose-Estimation-Dataset-Generator/`

## To Run:

Blenderproc 2.7.0

run this: `blenderproc run "./Dataset Generator/script.py"`

If you are running for the first time, enter 'y' when the program prompts you to generate the MANO hand models. They are needed to generate the dataset.


## To Debug:

run this: `blenderproc debug "./Dataset Generator/script.py"`
click on the "Run Blenderproc" button in Blender
