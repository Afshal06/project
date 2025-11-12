def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

def main():
    # Load model, dataloader, and device
    model = ...  # Load your trained model
    dataloader = ...  # Load your validation dataloader
    device = ...  # Set your device (CPU or GPU)

    # Evaluate the model
    loss, accuracy = evaluate_model(model, dataloader, device)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()