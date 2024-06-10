import json
import matplotlib.pyplot as plt
import os
import re
import sys
from data_classes import Event

try:
    filename = sys.argv[1]
except IndexError:
    # get the most recent output file...
    files = [f for f in os.listdir('.') if os.path.isfile(os.path.join('.', f))]
    pattern = r'^output.*\.json$'
    outputs = [file for file in files if re.match(pattern, file)]
    print(outputs)
    filename = outputs[-1]

with open(filename, 'r') as file:
    json_obj = json.load(file)
    output = [Event(**elem) for elem in json_obj]

# Create the visualization
x = []  # Scatterplot X values
y = []  # Scatterplot Y Values

# Loop over the data a second time
for event in output:
    x.append(event.timestamp)
    y.append(event.probe_point)

plt.figure(figsize=(14,4))
plt.title("Timeline Plot")
plt.xlim(output[0].timestamp, output[-1].timestamp)
plt.scatter(x, y)

plt.savefig(filename.replace("output", "timeline").replace("json", "svg"))