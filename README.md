# RCD_E3HDC
HDC platform 

System Requirements:<br>
Vivado 2022.2<br>
Python 3.10<br>
Conda / Miniconda (recommended)<br>

Setup steps:<br>
1. Create a new conda environment: conda create --name e3hdc python=3.10.12<br>
2. conda activate e3hdc<br>
3. Run: ./setup.sh<br>
The required python packages should now be installed for the environment.
4. Set the path and version of your vivado installation in "run_hdcgen.sh"

Test your installation by running: run_hdcgen.sh mnist_example/

General usage:
Create a folder containing a file called "hdc.py" which contains your training and test procedure with the functions "train" and "test".
The model and encoding should be saved to a folder called "model". An example script is provided in "mnist_example/".
Next to the training script "hdc.py" you require a configuration file, which should be named "config.json". It should look like this:
```json
{   
    "TRAIN"         : true,
    "FREQUENCY"     : 50,
    "LFSR"          : true,
    "DIMENSIONS"    : 1000,
    "FEATURES"      : 784,
    "NUM_LEVELS"    : 256,
    "NUM_CLASSES"   : 10,
    "SPARSE"        : true,
    "BOARD"         :"PYNQ-Z2",
    "PROJECT_NAME"  :"test"
}
```

Currently, only LFSR and sparseLFSR is supported. We are working on extending also to the standard base level encoding implementation.
