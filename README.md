# LipSyncr

LipSyncr is a *lip reading web application designed to accurately decipher spoken words from video footage*. Built upon the powerful LipNet model, this application employs advanced deep learning techniques to analyze and interpret lip movements with remarkable precision.


![Screenshot 2023-05-20 020423](https://github.com/SARIT42/lipsyncr/assets/77446629/f7ec4591-a643-4800-9cbb-b44ebb02b297)



### Dataset 
The dataset used for training the model is a subset of the [Grid Corpus Dataset](https://spandh.dcs.shef.ac.uk//gridcorpus/) .

To Download please run the following line of code in your terminal:
```
bash GridCorpus-Downloader.sh FirstSpeaker SecondSpeaker
```
where FirstSpeaker and SecondSpeaker are integers for the number of speakers to download

> NOTE: Speaker 21 is missing from the GRID Corpus dataset due to technical issues.

### Tech Stack Used
1. Python-Tensorflow-Keras -> data preparation, pipeline, model training & testing.
2. Streamlit -> web application.
3. LipNet -> lip reading model architecture idea.
4. ffmpeg -> video file format conversion
6. opencv -> video capture and frames processing.







