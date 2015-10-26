import chainer.functions as F
from chainer import Variable, FunctionSet
from contrastive import contrastive


class SiameseNetwork(FunctionSet):

    def __init__(self):
        super(SiameseNetwork, self).__init__(
            conv1=F.Convolution2D(1, 20, ksize=5, stride=1),
            conv2=F.Convolution2D(20, 50, ksize=5, stride=1),
            fc3=F.Linear(800, 500),
            fc4=F.Linear(500, 10),
            fc5=F.Linear(10, 2),
        )

    def forward_once(self, x_data, train=True):
        x = Variable(x_data, volatile=not train)

        h = F.max_pooling_2d(self.conv1(x), ksize=2, stride=2)
        h = F.max_pooling_2d(self.conv2(h), ksize=2, stride=2)
        h = F.relu(self.fc3(h))
        h = self.fc4(h)
        y = self.fc5(h)

        return y

    def forward(self, x0, x1, label, train=True):
        y0 = self.forward_once(x0, train)
        y1 = self.forward_once(x1, train)
        label = Variable(label, volatile=not train)

        return contrastive(y0, y1, label)
