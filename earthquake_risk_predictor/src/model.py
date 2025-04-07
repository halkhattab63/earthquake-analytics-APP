import torch
import torch.nn as nn
import torch.nn.functional as F

class EarthquakeModel(nn.Module):
    """
    Neural Network for Earthquake Severity Classification and Regression (Magnitude, Depth)
    Combines shared feature extraction, classification head, and regression head with output constraints.
    """
    def __init__(self, input_dim, num_classes=5):
        super(EarthquakeModel, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )

        # Classification head (multi-class severity levels)
        self.class_head = nn.Linear(64, num_classes)

        # Regression head with Softplus activation to ensure positive outputs
        self.reg_head = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        class_out = self.class_head(x)
        reg_out = self.reg_head(x)
        return class_out, reg_out
