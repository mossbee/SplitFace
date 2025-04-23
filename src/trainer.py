class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, num_epochs, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.scheduler = scheduler

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        return total_loss / len(self.train_loader), accuracy

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return total_loss / len(self.val_loader), accuracy

    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()

            if self.scheduler:
                self.scheduler.step()

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')