import torch
import torch.nn as nn
import torch.optim as optim

import config
from dataset import get_dataloaders
from models.cnn import SmallCNN
from utils.training_utils import train_one_epoch, evaluate


def main():
    print(f"Using device: {config.DEVICE}")

    train_loader, test_loader = get_dataloaders()

    model = SmallCNN(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    best_test_acc = 0.0

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=config.DEVICE
        )

        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=config.DEVICE
        )

        print(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Saved best model to {config.MODEL_SAVE_PATH}")

    print(f"\nTraining complete. Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    main()
