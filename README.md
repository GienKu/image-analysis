# Image Analysis Project

This project, implemented in C++ using OpenCV, serves as a learning exercise in image analysis and object recognition. The focus is on understanding concepts rather than optimization.

## Features

- **Image Binarization**: Converts grayscale images to binary format using simple thresholding.
- **Object Indexing**: Labels each object in the binary images with a unique identifier.
- **Feature Computation**:
  - **Moments**: Statistical measures related to shape and pixel intensity.
  - **Mass Moments**: Quantifies mass distribution within an object.
  - **Circumference**: Measures the object's perimeter.
  - **Area**: Computes the object's area.
- **Object Classification**: Classifies objects into groups based on computed features.
