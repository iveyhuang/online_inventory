# Online Inventory
This project is a simulation program for an online inventory management model

## Intsall Required Package with conda
1. To switch to the path of the file, you can use the `cd` command in the terminal. For example, if the download path of the file is `~/desktop/online_inventory`, you can run the command `cd ~/desktop/online_inventory` in the terminal to switch to that directory.

2. If you’re using a PC, you can create a virtual environment named `env_online` and install the required dependencies using the command `conda create –name env_online –file requirements.txt`. However, if you’re using a Mac with an M1 processor, you’ll need to run the command `conda create –name env_online –file requirements_m1_mac.txt` instead.

3. Once the virtual environment is created, you can activate it using the command `conda activate env_online`.

4. Finally, you can run the `main.py` file using the command `python main.py` while the virtual environment is activated.

5. If you don’t have Conda installed on your computer, you can refer to the instructions on how to [intsall Miniforge](https://equatorial-marlin-edd.notion.site/Install-Miniforge-on-Mac-of-M-chips-Windows-ec7d87d8c6494cca83681c5cbf9a3ac4).

## Use Google Colab
1. If you prefer not to download and install any development environment on your computer, there is an alternative option available. You can access the [main.ipynb](./colab/main.ipynb) file, which will allow you to view and edit the file without the need for additional package. 
2. To run the program, please click on the `Open In Colab` button. This will enable you to access and execute the code.
<img width="1087" alt="image" src="https://user-images.githubusercontent.com/114122443/225371274-0b043ca0-7170-4b42-b011-393b7d2386b4.png">
