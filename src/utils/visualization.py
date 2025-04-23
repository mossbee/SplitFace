def plot_attention_map(image, attention_map):
    import matplotlib.pyplot as plt
    import numpy as np

    # Normalize the attention map
    attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
    
    # Resize attention map to match the image size
    attention_map = np.resize(attention_map, (image.shape[0], image.shape[1]))

    # Create a heatmap
    plt.imshow(image, alpha=0.5)
    plt.imshow(attention_map, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

def visualize_predictions(images, predictions, titles=None):
    import matplotlib.pyplot as plt

    n = len(images)
    plt.figure(figsize=(15, 5))
    
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i] if titles else f'Pred: {predictions[i]}')
        plt.axis('off')
    
    plt.show()