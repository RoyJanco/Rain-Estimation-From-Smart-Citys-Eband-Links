# Rain-Estimation-From-Smart-Citys-Eband-Links
Python code for the short links empirical model and the RNN based network for rainfall estimation from smart city's E-band links, The network is based on PyTorch framework.

# Projects Structure
1. Rainfall estimation using the empirical short links model.
2. Wet/Dry classification and rainfall estimation from RNN based model.

# Dataset
This repository includes an example of a small dataset consisting of RSL measurments of five hops provided by SMBIT. LTD.
The rain gauge measurements are provided by The Robert H Smith Faculty of Agriculture, Food and Environment (Rehovot), The Hebrew University of
Jerusalem. http://www.meteo-tech.co.il/faculty/faculty_periodical.asp?client=1

# Usage
Example of Wet/Dry classification and rainfall estimation of both the empirical short links model and the RNN based model is available in the following notebook:
https://github.com/RoyJanco/Rain-Estimation-From-Smart-Citys-Eband-Links/blob/main/rain_estimation_demo.ipynb


# Models
In this project we supply a set of trained networks in our Models directory, these networks are trained on our own dataset which is not publicly available.
The Model directory contains RNN based networks for Wet/Dry classification and rainfall estimation. It includes models trained on different experiments:
* GeneralModels - contains four models with different configurations.
* NumLinks - contains three models trained on different number of links.
* TimePeriods - contains four models trained on different sizes of datasets.

ShortLinksModels directory contains the short links model parameters estimated from different sizes of datasets and from different links.

# Contributing
If you find a bug or have a question, please create a GitHub issue.



# Publications

Please cite the following paper if you found our work useful. Thanks!
>[1] R. Janco, J. Ostrometzky and H. Messer, “Rain Estimation from Smart City’s E-band Links,” 2022 IEEE 14th Image, Video, and Multidimensional Signal Processing Workshop (IVMSP), 2022, pp. 1-5, doi: 10.1109/IVMSP54334.2022.9816243.
```
@inproceedings{janco2022rain,
  title={Rain Estimation from Smart City’s E-band Links},
  author={Janco, Roy and Ostrometzky, Jonatan and Messer, Hagit},
  booktitle={2022 IEEE 14th Image, Video, and Multidimensional Signal Processing Workshop (IVMSP)},
  pages={1--5},
  year={2022},
  organization={IEEE}
}
```
Also this work is based on the following papers:

[2] Habi, Hai Victor and Messer, Hagit. "Wet-Dry Classification Using LSTM and Commercial Microwave Links"

[3] Habi, Hai Victor and Messer, Hagit. "RNN MODELS FOR RAIN DETECTION"

[4] Habi, Hai Victor. "Rain Detection and Estimation Using Recurrent Neural Network and Commercial Microwave Links"

[5] M. Schleiss and A. Berne, “Identification of dry and rainy periods usingtelecommunication microwave links,”IEEE Geoscience and RemoteSensing Letters, vol. 7, no. 3, pp. 611–615, 2010

If you found one of those methods papers useful please cite.
