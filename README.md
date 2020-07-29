# Computer Pointer Controller

In this project, the task is based on a gaze detection model to control the mouse pointer of the computer. The module state on the Intel model usage and will be applying the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly. This project will demonstrate the ability to run multiple models in the same machine and coordinate the flow of data between those models.

### How it works
You will be using the InferenceEngine API from Intel's OpenVino ToolKit to build the project. The gaze estimation model requires three inputs:

 - The head pose
 - The left eye image
 - The right eye image.
To get these inputs, you will have to use three other OpenVino models:

1. Face Detection
2. Head Pose Estimation
3. Facial Landmarks Detection.

The Pipeline
You will have to coordinate the flow of data from the input, and then amongst the different models and finally to the mouse controller. The flow of data will look like this:

[ ](media/pipeline.png)

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.




## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

### Reference 

1. OpenVINO Toolkit [Link](https://docs.openvinotoolkit.org/2020.1/index.html)
2. OpenCV tutorials [Link](https://www.pyimagesearch.com/2018/07/19/opencv-tutorial-a-guide-to-learn-opencv/)
3. D. W. Hansen and Q. Ji, “In the eye of the beholder: A survey of modelsfor eyes and gaze,” IEEE Trans. Pattern Anal. Mach. Intell, vol. 32, p.478500, Mar. 2010
4. D. J. McFarland, D. J. Krusienski, W. A.Sarnackia, W.A., and J. R.Wolpaw, “Emulation of computer mouse control with a noninvasivebraincomputer interface,” Journal of neural engineering, vol. 5, no. 2,p. 101, 2008.
5. A. Al-Rahayfeh and M. Faezipour, “Eye tracking and head movement de-tection: A state-of-art survey,” IEEE Journal of Translational Engineeringin Health and Medicine, vol. 1, pp. 2 100 212–2 100 212, 2013
6. Inference Engine API [Docs](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_ie_bridges_python_docs_api_overview.html)
7. Model Documentation [Link](https://docs.openvinotoolkit.org/latest/omz_models_intel_index.html)
8. D. Back, “Neural Network Gaze Tracking using Web Camera.,”
Linkping University, MS Thesis 2005.
9. Sixth Sense Technology [Wiki](https://en.wikipedia.org/wiki/SixthSense)