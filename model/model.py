import torch
import torch.optim as optim
import numpy as np

class ExplainableModel(torch.nn.Module):
    
    def __init__(self, input_dim, config):
        super(ExplainableModel, self).__init__()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, config['model']['input_layer']).float(),
            torch.nn.ReLU(),
        )
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Linear(config['model']['input_layer'], config['model']['hidden_layer_1']).float(),
            torch.nn.ReLU(),
            torch.nn.Linear(config['model']['hidden_layer_1'], config['model']['hidden_layer_2']).float(),
            torch.nn.ReLU(),
        )
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(config['model']['hidden_layer_2'], 1).float(),
            torch.nn.Sigmoid(),
        )

        self.__init_weights(config['model']['init_range'])

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

    def __init_weights(self, init_range):
        init_range = init_range
        self.input_layer[0].bias.data.zero_()
        self.input_layer[0].weight.data.uniform_(-init_range, init_range)
        self.hidden_layer[0].bias.data.zero_()
        self.hidden_layer[0].weight.data.uniform_(-init_range, init_range)
        self.output_layer[0].bias.data.zero_()
        self.output_layer[0].weight.data.uniform_(-init_range, init_range)

    def fit(self, x_train, y_train, x_val, y_val, config):
        self.train()

        criterion = torch.nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), config['model']['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['model']['step_size'], config['model']['gamma'])

        best_loss_val = float('inf')
        patience_counter = 0

        for i in range(config['model']['epochs']):
            y_pred = self.forward(torch.FloatTensor(x_train))
            loss = criterion(y_pred.squeeze(-1), torch.FloatTensor(y_train))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            self.eval()
            with torch.no_grad():
                y_pred_val = self.forward(torch.FloatTensor(x_val))
                loss_val = criterion(y_pred_val.squeeze(-1), torch.FloatTensor(y_val))

            # Early stopping
            if best_loss_val - loss_val > 1e-4:
                best_loss_val = loss_val
                patience_counter = 0  
            else:
                patience_counter += 1
                if patience_counter >= config['model']['early_stopping_patience']:
                    print("Early stopping triggered")
                    break
            self.train()
            print(f"Epochs {i+1}/{config['model']['epochs']} --> Training Loss: {loss.item():.4f}, Validation Loss: {loss_val.item():.4f}")
    
    def evaluate(self, x, y):
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(torch.FloatTensor(x))
            y_pred = torch.round(y_pred.squeeze(-1))

            correct = (y_pred == torch.FloatTensor(y)).sum().item()
            total   = len(y)
            accuracy = correct / total

        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy
    
    def predict(self, x):
        return self.forward(torch.FloatTensor(x))
