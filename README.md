This module is designed to perform a full (almost)-automatic analysis of the images obtained within the Declic experiments. 

It is heavily based on the PhD work of Jorge Pereda.

HOW TO INSTALL

* go into the directory with the setup.py file.
* >>> python setup.py install


...


HOW TO USE THE PACKAGE

    >>> from Deat import Deat
    >>> sequence = DEAT(path = pathToFolderWithImages, result_path = pathToResults\
                        start_tick=967938780, end_tick = 967950780, moteur = 967570127)
    >>> sequence.visu(0)
    >>> sequence.make_all_analysis(skip = 5)
    
    additional optionnal parameters for initialising sequence:
        * result_path
        * result_path
        * TNSP_file
        * wafer_file
        * cluster_file
        * pixel_size (default = 7.2)
        * moteur (default: 0)
        * start_tick (default: 0)
        * end_tick (default: inf)
        * eps (default: 8)
        * min_sample (default: 4)
        * verbose (default: False)
        * plot (default: False)
        * V (default: 2)
        * k_vel (default: 2)