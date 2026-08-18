import matplotlib.pyplot as plt

# Example accuracies (replace with your actual values)
accuracy_on_training_data = 0.50
accuracy_on_testing_data = 0.50

# Data for plotting
accuracies = [accuracy_on_training_data, accuracy_on_testing_data]
categories = ["Training", "Testing"]

# Plotting
plt.bar(categories, accuracies, color=['green', 'blue'])
plt.ylim(0.1, 0.9)  # Adjust the y-axis range if needed
plt.title("Training vs Testing Accuracy")
plt.ylabel("Accuracy")
plt.show()
