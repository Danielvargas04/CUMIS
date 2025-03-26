# Seismic Detection Across the Solar System

We developed an algorithm to accurately detect earthquake onset times from seismic signals, combining band-pass filtering, normalization, and the STA/LTA method to extract the signal's characteristic function. This function is smoothed using Fourier transforms for clearer identification of seismic events, which are then validated through spectral analysis. Our solution is adaptable for datasets from Earth, Mars, and the Moon, with specific adjustments for each environment, such as filtering frequencies and STA/LTA values. It efficiently detects multiple seismic events within a single signal and assigns a reliability score to each detection, with 98.7% accuracy and in less than 1 sec***. A key feature of the project is the development of an app that allows users to easily visualize the results by date, making seismic monitoring more accessible for scientists and authorities alike. This application has strong social relevance, as it provides an intuitive tool for early detection and analysis of seismic activity. Built entirely in Python, the project integrates various libraries and repositories to offer a powerful and flexible solution for global seismic research.

For details: [link](https://www.spaceappschallenge.org/nasa-space-apps-2024/find-a-team/cumis/?tab=project)
*** On the page, we have made a mistake, actually the data processing time of a single signal takes less than 1 sec. This error also appears in the section "Workflow Summary"

# HOW TO USE IT

This project processes seismic data using several Python libraries. Follow the instructions below to set up the environment and run the project.

## Requirements

- Python 3.x
- Internet connection to download the dataset

## Setup Instructions

### Step 1: Download the Data

1. Download the data by following this [link](https://drive.google.com/file/d/1Ga8_bZl1tN9ltSwH-2fwNPp3QREHnh2D/view?usp=sharing).
2. Extract the ZIP file in your project directory.

### Step 2: Install the Dependencies

To install the required Python dependencies, follow these steps:

1. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv myenv
   source myenv/bin/activate  # For Linux/MacOS
   myenv\Scripts\activate      # For Windows
2. Install the required libraries using the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt

### Step 3: Run the Project

Once you have the data and the dependencies set up, you can run the project:

```bash
python3 main.py
