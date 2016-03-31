# Sound Classification Tool
## Created by Adam Varga, 2015

This tool can be used for naïve sound classification. It extracts MFCC coefficients from the sound files, and applies different state-of-the art supervised classification 
methods on them. The number of MFCC-coefficients should be adjusted according to the characteristics of the data at hand by the `num_coeffs` parameter of the functions in `extract_features.py`.

### Usage
The sound files in WAV format should be placed in the `wav_samples` folder. All information should be coded into the file name and the info fields should be separated by dashes, the last field 
being the class label. Additional information can also be included; in the experiments this tool was used for age and gender have been used as extra features along with the MFCC vectors coded by the
penultimate and ante-penultimate fields in the filename. E. g. the filename `sample-29-male-shout.wav` would indicate an instance with class label "shout", being uttered by a 29-year-old male.

Sound files should be provided by the user; the claasification methods can be used for any sort of sound classification by slight modification of the code and free parameters. Train-test splitting is done
automatically as a 70% percentage split by default. 
