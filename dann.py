from torch import nn
from revgrad import RevGrad

class EEGDANN(nn.Module):
    def __init__(self):
        super(EEGDANN, self).__init__()
        self.revgrad = RevGrad()
        self.feat_extr = nn.Sequential(
            nn.Linear(310, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        self.label_classify = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.domain_classify = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 14)
        )

    def forward(self, input):
        feature = self.feat_extr(input)
        label_pred = self.label_classify(feature)
        domain_pred = self.domain_classify(self.revgrad(feature))
        return label_pred, domain_pred
