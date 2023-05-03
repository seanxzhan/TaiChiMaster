# Tai Chi Master

![teaser1](/images/1.png)

## About

Tai Chi Master is a tool created by Edward Xing and Xiao (Sean) Zhan to measure a user's accuracy in performing Tai Chi 24. It provides an objective measurement based on the joint angles of key body parts. Accuracy is computed by comparing the joint angles of the user to that of the ground truth based on Master Amin Wu.

## Study

Tai Chi Master was developed with the intent to explore whether accuracy affects the benefits of Tai Chi. To measure the benefits of Tai Chi, heart rate was used as an objective measurement, and chi was used as a subjective measurement. To determine if accuracy had an effect, the correlation between accuracy and each metric was calculated over multiple observations. 

From the trials conducted, there did not seem to be any correlation. Therefore, accuracy does not seem to impact the benefits of Tai Chi, implying that simply doing Tai Chi is more important than doing it well.

## Presentation
Slides containing more information are available [here](https://docs.google.com/presentation/d/17qAxEklpiHCAzNIXFjC4eY1SAIED22zMmu4B1uLttX8/edit?usp=sharing).

## Running
- Install dependencies with `pip install -r requirements.txt`
- This project uses OpenPose to estimate poses. Please install [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
- In `run_online_24_poses.py`, replace
  - the openpose python path with your path. 
  - the openpose models path with your path.
- Try out your pose accuracy against some sample poses by running `python run_online_24_poses.py`
- Here's an example of what it looks like:

![teaser1](/images/3.png)