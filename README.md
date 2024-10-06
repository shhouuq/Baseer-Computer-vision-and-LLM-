# Baseer Application 

![logo](https://github.com/user-attachments/assets/2855db3e-f722-4f9f-b194-74d969a91f70)


##### An application that uses Computer vision and LLM to analyze traffic during large events in Saudi Arabia.

Our project tackles the issue of traffic congestion, especially
during significant events such as festivals and sports
competitions. With the rapid expansion of cities in Saudi
Arabia and the increasing frequency of large gatherings,
efficient traffic management is crucial. We plan to implement
computer vision techniques to identify and forecast different
traffic conditions, allowing for real-time modifications to
traffic systems. The proposed approach employs deep
learning models and technuiques, Specifically Computer vision and LLM to monitor and improve traffic flow, ultimately
aiming to alleviate congestion and enhance the overall traffic
experience during major events.

## Table of Contents

- [Data](#data)
- [Structure](#structure)
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)

## Data

The data used in this project were gathered manually from platforms and social media across the internet. The data reched to 1200 images seperated into 3 categories: 
- Caused traffic (400 img)
- Regular traffic (400 img)
- No traffic (400 img)

## Structure

Our project uses computer vision and natural language
processing. It has three main steps: training a model with
images, analyzing videos in real-time, and integrating an AI
chatbot.

As mentioned before, our dataset included over 1,200 images
that were manually sorted into three categories: "Caused traffic," "Regular traffic," and "No traffic" These
images were used to train a convolutional neural network
(CNN) with Keras. Once a baseline accuracy was achieved,
video data was added for real-time analysis. A custom Flask
application processed the videos frame-by-frame with a finetuned YOLO (You Only Look Once) model. The results were
displayed with OpenCV annotations to show different types of
congestion. Lastly, a LangChain-based chatbot, using the
llama3-8b-8192 language model, was added to give contextaware answers to user questions about traffic patterns.
These methods creates a strong system for accurately
monitoring and responding to traffic conditions that will help
to maintain during major events.

<img width="1003" alt="Screenshot 1446-04-04 at 2 49 05 AM" src="https://github.com/user-attachments/assets/b835172b-03db-4fd8-8ed5-11628429c665">


<img width="1025" alt="Screenshot 1446-04-04 at 2 50 36 AM" src="https://github.com/user-attachments/assets/c6d81493-59ba-4c12-9419-6a4730087009">




## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shhouuq/Baseer-Computer-vision-and-LLM-.git

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Usage

[click here](https://drive.google.com/drive/folders/1cwcCq1WSys0HbrQnpHc-rHRpwfuAAOFL?usp=drive_link) to load the classification model.

## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. Make sure to update the tests as appropriate.
