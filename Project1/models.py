from torch import nn
from torch import optim
from torch import Tensor
from torch.nn import functional as F

############Models without weight sharing
class C1L2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 208, kernel_size=3,groups=2)
        self.fc1 = nn.Linear(3328, 20)
        self.fc2 = nn.Linear(20, 2)
        self.aux_linear = nn.Linear(20, 20)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        aux_output = F.softmax(self.fc1(x.view(-1, 3328)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 3328)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output

class C1L3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3,groups=2)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 2)
        self.aux_linear = nn.Linear(20, 20)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = F.relu(self.fc2(x))
        aux_output = F.softmax(x, dim=1)
        output = F.softmax(self.fc3(x),dim=1)
        return output, aux_output

class C1L5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 20, kernel_size=3,groups=2)
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80,40)
        self.fc4 = nn.Linear(40,20)
        self.fc5 = nn.Linear(20, 2)
        self.aux_linear = nn.Linear(20, 20)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(self.fc1(x.view(-1, 320)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        aux_output = F.softmax(x, dim=1)
        output = F.softmax(self.fc5(x),dim=1)
        return output, aux_output
    
class C2L2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 48, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(48, 240, kernel_size=3)
        self.fc1 = nn.Linear(960, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        aux_output = F.softmax(self.fc1(x.view(-1, 960)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 960)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output
    
class C2L3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, 60)
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        aux_output = F.softmax(self.fc1(x.view(-1, 512)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output
    
class C3L2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 22, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(22, 44, kernel_size=3)
        self.conv3 = nn.Conv2d(44, 88, kernel_size=3)
        self.fc1 = nn.Linear(1408, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        aux_output = F.softmax(self.fc1(x.view(-1, 1408)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 1408)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output
    
class C4L2_bis(nn.Module):#bis got switched, be careful
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 12, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3)
        self.conv4 = nn.Conv2d(48, 96, kernel_size=3)
        self.fc1 = nn.Linear(864, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(F.max_pool2d(self.conv4(x), kernel_size=2, stride=2))
        aux_output = F.softmax(self.fc1(x.view(-1, 864)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 864)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output

class C4L2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3,groups=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.fc1 = nn.Linear(1152, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        aux_output = F.softmax(self.fc1(x.view(-1, 1152)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 1152)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output
    
############Models with weight sharing
class C1L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 104, kernel_size=3)
        self.fc1 = nn.Linear(3328, 20)
        self.fc2 = nn.Linear(20, 2)
        self.aux_linear = nn.Linear(20, 20)

    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=3, stride=3))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=3, stride=3))
        x = torch.cat((x1,x2), 1)
        aux_output = F.softmax(self.fc1(x.view(-1, 3328)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 3328)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output

class C1L3WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 20)
        self.fc3 = nn.Linear(20, 2)
        self.aux_linear = nn.Linear(20, 20)

    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=3, stride=3))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=3, stride=3))
        x = torch.cat((x1,x2), 1)
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = F.relu(self.fc2(x))
        aux_output = F.softmax(x, dim=1)
        output = F.softmax(self.fc3(x),dim=1)
        return output, aux_output
    
class C1L5WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80,40)
        self.fc4 = nn.Linear(40,20)
        self.fc5 = nn.Linear(20, 2)
        self.aux_linear = nn.Linear(20, 20)

    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        x1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=3, stride=3))
        x2 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=3, stride=3))
        x = torch.cat((x1,x2), 1)
        x = F.relu(self.fc1(x.view(-1, 320)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        aux_output = F.softmax(x, dim=1)
        output = F.softmax(self.fc5(x),dim=1)
        return output, aux_output
    
class C2L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3,groups=1)
        self.conv2 = nn.Conv2d(24, 120, kernel_size=3)
        self.fc1 = nn.Linear(960, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        x1_1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1_1), kernel_size=2, stride=2))
        x2_1 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2_1), kernel_size=2, stride=2))
        x = torch.cat((x1,x2), 1)

        aux_output = F.softmax(self.fc1(x.view(-1, 960)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 960)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output
    
class C2L3WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3)
        self.fc1 = nn.Linear(512, 60)
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        x1_1 = F.relu(F.max_pool2d(self.conv1(picture1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1_1), kernel_size=2, stride=2))
        x2_1 = F.relu(F.max_pool2d(self.conv1(picture2), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2_1), kernel_size=2, stride=2))
        
        x = torch.cat((x1,x2), 1)
        aux_output = F.softmax(self.fc1(x.view(-1, 512)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = F.relu(self.fc2(x))
        output = F.softmax(self.fc3(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output
    
class C3L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 11, kernel_size=3)
        self.conv2 = nn.Conv2d(11, 22, kernel_size=3)
        self.conv3 = nn.Conv2d(22, 44, kernel_size=3)
        self.fc1 = nn.Linear(1408, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        x1_1 = F.relu(self.conv1(picture1))
        x1_2 = F.relu(self.conv2(x1_1))
        x1 = F.relu(F.max_pool2d(self.conv3(x1_2), kernel_size=2, stride=2))
        x2_1 = F.relu(self.conv1(picture2))
        x2_2 = F.relu(self.conv2(x2_1))
        x2 = F.relu(F.max_pool2d(self.conv3(x2_2), kernel_size=2, stride=2))

        x = torch.cat((x1,x2), 1)
        aux_output = F.softmax(self.fc1(x.view(-1, 1408)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 1408)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output
    
class C4L2WS_bis(nn.Module):#bis got changed
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3)
        self.conv3 = nn.Conv2d(12, 24, kernel_size=3)
        self.conv4 = nn.Conv2d(24, 48, kernel_size=3)
        self.fc1 = nn.Linear(864, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        
        x1_1 = F.relu(self.conv1(picture1))
        x1_2 = F.relu(self.conv2(x1_1))
        x1_3 = F.relu(self.conv3(x1_2))
        x1 = F.relu(F.max_pool2d(self.conv4(x1_3), kernel_size=2, stride=2))
        x2_1 = F.relu(self.conv1(picture2))
        x2_2 = F.relu(self.conv2(x2_1))
        x2_3 = F.relu(self.conv3(x2_2))
        x2 = F.relu(F.max_pool2d(self.conv4(x2_3), kernel_size=2, stride=2))
        x = torch.cat((x1,x2), 1)

        aux_output = F.softmax(self.fc1(x.view(-1, 864)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 864)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output

class C4L2WS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(1152, 20)
        self.fc2 = nn.Linear(20, 2)

        
    def forward(self, x):
        picture1 =x.narrow(1, 0, 1)
        picture2 =x.narrow(1, 1, 1)
        x1_1 = F.relu(self.conv1(picture1))
        x1_2 = F.relu(F.max_pool2d(self.conv2(x1_1), kernel_size=2, stride=2))
        x1_3 = F.relu(self.conv3(x1_2))
        x1 = F.relu(self.conv4(x1_3))
        x2_1 = F.relu(self.conv1(picture2))
        x2_2 = F.relu(F.max_pool2d(self.conv2(x2_1), kernel_size=2, stride=2))
        x2_3 = F.relu(self.conv3(x2_2))
        x2 = F.relu(self.conv4(x2_3))
        x = torch.cat((x1,x2), 1)

        aux_output = F.softmax(self.fc1(x.view(-1, 1152)), dim=1)
        x = F.relu(self.fc1(x.view(-1, 1152)))
        output = F.softmax(self.fc2(x), dim=1)
        aux_output = F.softmax(x, dim=1)
        return output, aux_output