import numpy as np
import matplotlib.pyplot as plt

# Paste your confusion matrix output here
cm_text = """
actual\pred  1000-1199  1200-1399  1400-1599  1600-1799  1800-1999  2000-2199  2200-2399  2400-2599
1000-1199          289       5942      11069       2354         94          2          0          0
1200-1399           31       4680      25289      12543        543          0          0          0
1400-1599           14       2344      34372      41318       4629         62          0          0
1600-1799            0        356      20291      70356      20767        640          0          0
1800-1999            0         57       5168      47993      41396       2739          4          0
2000-2199            0          2        480      12977      26573       4199         52          0
2200-2399            0          0         66       2226       7932       2531         90          0
2400-2599            0          0          5        289       1425        556          7          0
"""

def parse_confusion_matrix(text):
  lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
  # Extract labels from the header (skipping 'actual\pred')
  labels = lines[0].split()[1:]
  
  matrix = []
  for line in lines[1:]:
    parts = line.split()
    # Skip the row label (parts[0]) and parse the rest as integers
    matrix.append([int(val) for val in parts[1:]])
    
  return np.array(matrix), labels

def plot_heatmap(confusion_arr, labels, title='Confusion Matrix Heatmap'):
  fig, ax = plt.subplots(figsize=(10, 8))
  # Using 'Blues' to match get_bin_accuracy.py
  cax = ax.imshow(confusion_arr, interpolation='nearest', cmap='Blues')
  fig.colorbar(cax)
  
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels, rotation=45, ha='right')
  ax.set_yticklabels(labels)
  
  # Add text annotations with contrast awareness
  thresh = confusion_arr.max() / 2.
  for i in range(len(labels)):
    for j in range(len(labels)):
      color = "white" if confusion_arr[i, j] > thresh else "black"
      ax.text(j, i, format(confusion_arr[i, j], ','), 
              ha='center', va='center', color=color)
      
  ax.set_xlabel('Predicted')
  ax.set_ylabel('Actual')
  ax.set_title(title)
  fig.tight_layout()
  plt.show()

if __name__ == "__main__":
  matrix, labels = parse_confusion_matrix(cm_text)
  plot_heatmap(matrix, labels)