# RCEHDC
HDC platform 

System Requirements:<br>
Vivado 2022.2<br>
Python 3.10<br>
Conda / Miniconda (recommended)<br>

Setup steps:<br>
1. Create a new conda environment: conda create --name rcehdc python=3.10.12<br>
2. conda activate rcehdc<br>
3. Run: ./setup.sh<br>
The required python packages should now be installed for the environment.
4. Set the path and version of your vivado installation in "run_hdcgen.sh"

Test your installation by running: run_hdcgen.sh mnist_example/
You can find the generated .bit and .hwh file in the "release" folder of the directory.

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

A Vivado project with PROJECT_NAME will be created at the specified directory. For each step, a log file will be created. Some of the logs, related to the Vivado runs (synthesis and implementation), will be generated in the corresponding project directory, such as "synth_1".
In the rare case of out-of-context synthesis failing, the program might hang due to missing feedback from Vivado. The program can be safely exited in that case. We will soon add a check for this as well.

The usual hardware generation should take around 700 or 450 seconds on a standard office laptop, for LFSR and sparseLFSR respectively. If it takes significantly longer, please try again with different parameters or check the log files of the OOC runs.
