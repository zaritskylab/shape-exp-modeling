
# Project Title

![Project Image](path_to_your_image)

## Summary

This project is based on the article [Data-modeling the interplay between single cell shape, single cell protein expression, and tissue state](https://www.biorxiv.org/content/10.1101/2024.05.29.595857v1). The study combines spatial multiplexed single-cell imaging and machine learning to explore the intricate relationships between cell shape and protein expression within human tissues. The results highlight a universal bi-directional link between cell shape and protein expression across various cell types and disease states. This research opens new avenues for understanding cellular behavior and improving disease state predictions.

## Usage

First, let's import the necessary modules and process the data.

```python
# Import necessary modules
from ProcessData import process_data
import matplotlib.pyplot as plt

# Process the data
data = process_data('path/to/your/data')

# Display some processed data (example plot)
plt.figure(figsize=(10, 6))
plt.plot(data['feature1'], label='Feature 1')
plt.plot(data['feature2'], label='Feature 2')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Processed Data Features')
plt.legend()
plt.show()
```

### Model Training

Next, we'll train a model using the processed data.

```python
# Import the model and training functions
from models import YourModel
from train_eval import train_model

# Initialize the model
model = YourModel()

# Train the model
train_model(model, data)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(model.history['loss'], label='Training Loss')
plt.plot(model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```

### Model Evaluation

Finally, we'll evaluate the trained model and display the results.

```python
# Import evaluation functions
from train_eval import evaluate_model

# Evaluate the model
results = evaluate_model(model, data)

# Display evaluation results (example plot)
plt.figure(figsize=(10, 6))
plt.plot(results['true_values'], label='True Values')
plt.plot(results['predicted_values'], label='Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Model Evaluation Results')
plt.legend()
plt.show()
```

For more detailed examples and explanations, please refer to the [Shape2Exp_Demo.ipynb](Shape2Exp_Demo.ipynb) notebook included in this repository.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
