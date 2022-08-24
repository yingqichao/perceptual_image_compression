import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel=1):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, in_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=False, norm=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)



class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False, norm=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False, norm=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class SimplePatchGAN(nn.Module):
    def __init__(self):
        super().__init__()
        base_channel = 32
        self.conv1 = BasicConv(3, base_channel, kernel_size=3, stride=1, relu=True, norm=True)
        self.conv2 = BasicConv(base_channel, base_channel*2, kernel_size=3, stride=1, relu=True, norm=True)
        self.conv3 = BasicConv(base_channel*2, base_channel * 4, kernel_size=3, stride=1, relu=True, norm=True)
        self.conv4 = BasicConv(base_channel * 4, base_channel * 8, kernel_size=3, stride=1, relu=True, norm=True)
        # self.conv5 = BasicConv(base_channel * 8, base_channel * 16, kernel_size=3, stride=1, relu=True, norm=True)
        self.final_conv = BasicConv(base_channel * 8,1, kernel_size=1, stride=1, relu=False, norm=False)

    def forward(self,x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=0.5)
        x = self.conv2(x) #16
        x = F.interpolate(x, scale_factor=0.5)
        x = self.conv3(x)  #8
        x = F.interpolate(x, scale_factor=0.5)
        x = self.conv4(x)  # 4
        # x = F.interpolate(x, scale_factor=0.5)
        # x = self.conv5(x)  # 2
        x = self.final_conv(x)
        return x



class MIMOUNet(nn.Module):
    def __init__(self, num_res=8):
        super(MIMOUNet, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, norm=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, norm=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, norm=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        """
        注意：SCM没啥卵用，FAM也就一个AdaIn而已，AFF就是一个把三个不同尺度feature resize一下拼一起卷积，都没啥卵用
        :param x:
        :return:
        """
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        # outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        z = self.Decoder[0](z)
        # z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        # outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        # z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        # outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        # outputs.append(z+x)

        return x+torch.tanh(z)

class MIMOUNet_encoder(nn.Module):
    def __init__(self, num_res=8, scale=32, compression_rate=1.0):
        super(MIMOUNet_encoder, self).__init__()
        self.scale = scale/32.0
        self.compression_rate = compression_rate
        base_channel = 32

        original_scale = self.scale*self.scale*(0.25*0.25+0.5*0.5+1)/3
        # further_scaling = compression_rate/original_scale
        print(f"Current s: {original_scale}")
        # print(f"further scaling factor:{further_scaling}")

        # self.MLP1 = nn.Sequential(
        #     nn.Linear(int(32 * 32 * scale * scale), int(32 * 32 * scale * scale)),
        #     # nn.BatchNorm1d(int(32 * 32 * scale * scale)),
        #     nn.ELU(inplace=True),
        #     nn.Linear(int(32*32*scale*scale),int(32*32*scale*scale*further_scaling))
        # )
        # self.MLP2 = nn.Sequential(
        #     nn.Linear(int(16 * 16 * scale * scale), int(16 * 16 * scale * scale)),
        #     # nn.BatchNorm1d(int(16 * 16 * scale * scale)),
        #     nn.ELU(inplace=True),
        #     nn.Linear(int(16*16*scale*scale),int(16*16*scale*scale*further_scaling))
        # )
        # self.MLP3 = nn.Sequential(
        #     nn.Linear(int(8 * 8 * scale * scale), int(8 * 8 * scale * scale)),
        #     # nn.BatchNorm1d(int(8 * 8 * scale * scale)),
        #     nn.ELU(inplace=True),
        #     nn.Linear(int(8*8*scale*scale),int(8*8*scale*scale*further_scaling))
        # )
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, norm=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, norm=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, norm=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, out_channel=1),
            AFF(base_channel * 7, out_channel=1)
        ])

        self.conv1x1 = BasicConv(base_channel*4, 1, kernel_size=1, stride=1, relu=False, norm=False)

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        batchsize, *_ = x.shape

        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)


        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5*self.scale) if 0.5*self.scale!=1 else res1
        z21 = F.interpolate(res2, scale_factor=2*self.scale) if 2*self.scale!=1 else res2
        z42 = F.interpolate(z, scale_factor=2*self.scale) if 2*self.scale!=1 else z
        z41 = F.interpolate(z, scale_factor=4*self.scale) if 4*self.scale!=1 else z42
        res1 = F.interpolate(res1, scale_factor=1*self.scale) if 1*self.scale!=1 else res1
        res2 = F.interpolate(res2, scale_factor=1 * self.scale) if 1 * self.scale != 1 else res2
        z = F.interpolate(z, scale_factor=1 * self.scale) if 1 * self.scale != 1 else z

        res1 = self.AFFs[0](res1, z21, z41).view(batchsize, -1)
        res2 = self.AFFs[1](z12, res2, z42).view(batchsize,-1)
        z = self.conv1x1(z).view(batchsize,-1)

        outputs = torch.concat((res1,res2,z),dim=1)

        return outputs

class MIMOUNet_decoder(nn.Module):
    def __init__(self, num_res=8,scale=32,compression_rate=0.5):
        super(MIMOUNet_decoder, self).__init__()
        self.compression_rate = compression_rate
        base_channel = 32
        scale = scale/32

        original_scale = scale * scale * (0.25 * 0.25 + 0.5 * 0.5 + 1) / 3
        # further_scaling = compression_rate / original_scale
        print(f"Current s: {original_scale}")
        # print(f"further scaling factor:{further_scaling}")
        self.index1 = int(32 * 32 * scale * scale)
        self.index2 = int(16 * 16 * scale * scale)
        self.index3 = int(8 * 8 * scale * scale)

        # self.MLP1 = nn.Sequential(
        #     nn.Linear(self.index1, self.index1),
        #     # nn.BatchNorm1d(self.index1),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.index1,int(32*32*scale*scale))
        # )
        # self.MLP2 = nn.Sequential(
        #     nn.Linear(self.index2, self.index2),
        #     # nn.BatchNorm1d(self.index2),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.index2,int(16*16*scale*scale))
        # )
        # self.MLP3 = nn.Sequential(
        #     nn.Linear(self.index3, self.index3),
        #     # nn.BatchNorm1d(self.index3),
        #     nn.ELU(inplace=True),
        #     nn.Linear(self.index3,int(8*8*scale*scale))
        # )

        self.scale = 1 / scale
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, norm=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, norm=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, norm=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.conv1x1_4 = BasicConv(1, base_channel, kernel_size=1, stride=1, relu=False, norm=False)
        self.conv1x1_2 = BasicConv(1, base_channel*2, kernel_size=1, stride=1, relu=False, norm=False)
        self.conv1x1_1 = BasicConv(1, base_channel*4, kernel_size=1, stride=1, relu=False, norm=False)

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, input_tensor):
        batchsize, _ = input_tensor.shape
        res1 = input_tensor[:,:self.index1].view(-1, 1, int(32/self.scale),int(32/self.scale))
        res2 = input_tensor[:, self.index1:self.index1+self.index2].view(-1, 1, int(16/self.scale), int(16/self.scale))
        z = input_tensor[:, self.index1+self.index2:].view(-1, 1, int(8/self.scale), int(8/self.scale))

        res1 = self.conv1x1_4(res1)
        res2 = self.conv1x1_2(res2)
        z = self.conv1x1_1(z)
        if self.scale != 1:
            res1 = F.interpolate(res1, size=(32,32))
            res2 = F.interpolate(res2, size=(16,16))
            z = F.interpolate(z, size=(8,8))

        z = self.Decoder[0](z)
        z = self.feat_extract[3](z)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z = self.feat_extract[4](z)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        outputs = self.feat_extract[5](z)

        return torch.tanh(outputs)


class Simple_Class_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MIMOUNetPlus(nn.Module):
    def __init__(self, num_res = 20):
        super(MIMOUNetPlus, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, norm=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])

        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.ConvsOut = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, norm=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, norm=False, stride=1),
            ]
        )

        self.AFFs = nn.ModuleList([
            AFF(base_channel * 7, base_channel*1),
            AFF(base_channel * 7, base_channel*2)
        ])

        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

        self.drop1 = nn.Dropout2d(0.1)
        self.drop2 = nn.Dropout2d(0.1)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)

        z12 = F.interpolate(res1, scale_factor=0.5)
        z21 = F.interpolate(res2, scale_factor=2)
        z42 = F.interpolate(z, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        res2 = self.AFFs[1](z12, res2, z42)
        res1 = self.AFFs[0](res1, z21, z41)

        res2 = self.drop2(res2)
        res1 = self.drop1(res1)

        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs


def build_net(model_name):
    class ModelError(Exception):
        def __init__(self, msg):
            self.msg = msg

        def __str__(self):
            return self.msg

    if model_name == "MIMO-UNetPlus":
        return MIMOUNetPlus()
    elif model_name == "MIMO-UNet":
        return MIMOUNet()
    raise ModelError('Wrong Model!\nYou should choose MIMO-UNetPlus or MIMO-UNet.')

if __name__ == '__main__':
    model = MIMOUNet_encoder().cuda()
    input = torch.ones((1,3,64,64)).cuda()
    output = model(input)
    print(output.shape)
