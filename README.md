# Subway Surfers CNN

A LSTM-CNN based player for the game Subway Surfers, written in TensorFlow.
An Example of how to define a model, and a dataset is provided in configs folder, architectures.py, and the generated/data folder. 
<br/>

This currently uses a 3 convolution, and 3 pooling layer deep CNN, along with an LSTM after flattening the Convolution Network. The final model has around 750K parameters. Initially I tried to predict only using a CNN. Although the predictions using it were mostly correct, the predictions were too early. I initially tried to a manual approach by holding onto previous few frame predictions, but in the sprit of automation, I looked for other solutions. An pretty much did what I was doing manually.<br/>

This was trained on an agumented dataset of 160K images (80K original, mirrored vertically), downscaled to 170x82.
The data was manually collected by playing the game, while running a scaper in background. <br/>
The highest score the model was able to achieve was around 8K, which is impressive given that it was only trained for runs upto 10K. It could also beat 6K consistently. <br/>
The model has an accuracy of 86 percent, although this stat does not mean much, as the dataset is highly imbalanced, having most of the images as DO_NOTHING. I am undersampling the majority class, so I could achieve any accuracy I wanted by overfiting the model to always predict DO_NOTHING. <br/>
The main metric I was looking for is recall. Here I considered TP as how many times any action was required, and predicted. And FN as how many any action was required, and not taken. This is because taking a wrong action is the vast majority of times, harmless, but not taking an action usually means game over. <br/>
I achieved about .85 recall for my model. This is the confusion matrix.



<br/>
A pre-trained model is  provided in generated/output. <br/>
Unfortunately, the entire dataset (Of 80K images) is too big to upload on github. Feel free to contact me if you need it for some reason. <br/>

The model was trained on TensorFlow 2.10.1 is required, which is required to load it. An android emulator is also required. <br/>
Before running, the name of the emulator's window needs to be set as SCREEN_NAME in settings.py. <br/>
CAPTURE_LTRB_OFFSET also needs to be adjusted to make sure only the relevant part of screen in captured.
After these two changes are made, start the emulator and simply run: <br/>
python player.py <br/>
to run the player.
To train the model from configs, just run <br/>
python main.py <br/>
To capture custom data, start the game on emulator, then run. <br/>
python datagen.py <br/>
Dxcam library is required for this capturing or playing.