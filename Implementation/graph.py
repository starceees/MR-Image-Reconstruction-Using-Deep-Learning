#!/usr/bin/env python3
from graphviz import Digraph

def create_model_diagram():
    # Create a new directed graph; set output format to PDF.
    dot = Digraph(comment='Custom UNet2D Architecture and Data Pipeline', format='pdf')
    
    # Set layout direction to left-to-right for a horizontal diagram.
    dot.attr(rankdir='LR')
    
    # -----------------------------
    # Data Pipeline Nodes
    # -----------------------------
    dot.node('A', 'Input 3D Medical Volume\n(NIfTI Format)', shape='box', style='filled', color='lightblue')
    dot.node('B', 'Slice Extraction\n(axial slices)', shape='box', style='filled', color='lightgrey')
    dot.node('C', 'Normalization\n(min-max scaling)', shape='box', style='filled', color='lightgrey')
    dot.node('D', 'Data Augmentation\n(Random flips, rotations, zoom)', shape='box', style='filled', color='lightgrey')
    dot.node('E', 'Data Loader\n(Train/Val/Test Split)', shape='box', style='filled', color='lightgreen')
    
    # -----------------------------
    # Custom UNet2D Model Cluster
    # -----------------------------
    with dot.subgraph(name='cluster_Model') as c:
        c.attr(style='filled', color='lightyellow')
        c.attr(label='Custom UNet2D Model')
        c.node('F1', 'Encoder\n(Conv + ReLU + MaxPool)', shape='box')
        c.node('F2', 'Bottleneck\n(Conv + ReLU)', shape='box')
        c.node('F3', 'Decoder\n(Upsampling + Skip Connections)', shape='box')
        # Connect components within the model cluster
        c.edge('F1', 'F2')
        c.edge('F2', 'F3')
    
    # -----------------------------
    # Output Node
    # -----------------------------
    dot.node('G', 'Segmentation Mask\n(Class Prediction)', shape='box', style='filled', color='lightblue')
    
    # -----------------------------
    # Connect the Data Pipeline to the Model and Output
    # -----------------------------
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F1')  # Data loader feeds into the encoder stage of the model
    dot.edge('F3', 'G')  # Decoder outputs the segmentation mask
    
    # -----------------------------
    # (Optional) Add an annotation node
    # -----------------------------
    dot.node('H', 'Data Pipeline and Model Training', shape='note', style='dashed')
    dot.edge('H', 'A', style='dashed')
    
    # Render the graph to a PDF file and open it.
    dot.render('model_architecture_diagram', view=True)

if __name__ == '__main__':
    create_model_diagram()
