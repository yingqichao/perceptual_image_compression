import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from models.MIMOUNet import MIMOUNet_encoder, MIMOUNet_decoder, \
    MIMOUNet, Simple_Class_Net, SimplePatchGAN
from metrics import PSNR, postprocess
from models.loss import CWLoss
from models.class_models.vgg import VGG
from models.class_models.resnet import ResNet18
from models.class_models.googlenet import GoogLeNet
from utils import stitch_images, clamp_with_grad
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint', type=int, default=0.0, help='')
parser.add_argument('-compression_rate', type=float, default=0.4, help='')
parser.add_argument('-cls_weight', type=float, default=0.005, help='')
parser.add_argument('-thresh', type=int, default=28, help='')
args = parser.parse_args()
print(args)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 64
print("using CIFAR100")
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########### Settings ################
print_step, save_step = 50, 750
psnr_thresh = args.thresh # 28
cls_weight = args.cls_weight #0.005
compression_rate = args.compression_rate
print(f"scale is {compression_rate}")
#####################################
model_en = MIMOUNet_encoder(compression_rate=compression_rate).cuda()
model_de = MIMOUNet_decoder(compression_rate=compression_rate).cuda()
model_rec = MIMOUNet().cuda()
# net = Simple_Class_Net().cuda()
model_dis = SimplePatchGAN().cuda()
net1 = VGG('VGG19', num_classes=100).cuda()
net2 = ResNet18(num_classes=100).cuda()
# net = PreActResNet18()
net3 = GoogLeNet(num_classes=100).cuda()
# net = DenseNet121()

if args.checkpoint!=0:
    PATH = f'./vgg_{str(compression_rate * 10)}.pth'
    net1.load_state_dict(torch.load(PATH))
    PATH = f'./res_{str(compression_rate * 10)}.pth'
    net2.load_state_dict(torch.load(PATH))
    PATH = f'./google_{str(compression_rate * 10)}.pth'
    net3.load_state_dict(torch.load(PATH))
    PATH = f'./model_en_{str(compression_rate * 10)}.pth'
    model_en.load_state_dict(torch.load(PATH))
    PATH = f'./model_de_{str(compression_rate * 10)}.pth'
    model_de.load_state_dict(torch.load(PATH))
    PATH = f'./model_dis_{str(compression_rate * 10)}.pth'
    model_dis.load_state_dict(torch.load(PATH))
    PATH = f'./model_rec_{str(compression_rate * 10)}.pth'
    model_rec.load_state_dict(torch.load(PATH))

cw_loss = CWLoss(num_classes=100).cuda()
psnr = PSNR(255.0).cuda()
criterion = nn.CrossEntropyLoss().cuda()
l1_loss = nn.SmoothL1Loss().cuda()
bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()
optimizer_dis = optim.AdamW(model_dis.parameters(),
                                 lr=2e-4,betas=(0.9, 0.999), weight_decay=0.01)
optimizer_en = optim.AdamW(model_en.parameters(),
                                 lr=2e-4,betas=(0.9, 0.999), weight_decay=0.01)
optimizer_de = optim.AdamW(model_de.parameters(),
                                 lr=2e-4,betas=(0.9, 0.999), weight_decay=0.01)
optimizer_net1 = optim.AdamW(net1.parameters(),
                                 lr=1e-3,betas=(0.9, 0.999), weight_decay=0.01)
optimizer_net2 = optim.AdamW(net2.parameters(),
                                 lr=1e-3,betas=(0.9, 0.999), weight_decay=0.01)
optimizer_net3 = optim.AdamW(net3.parameters(),
                                 lr=1e-3,betas=(0.9, 0.999), weight_decay=0.01)
optimizer_rec = optim.AdamW(model_rec.parameters(),
                                 lr=2e-4,betas=(0.9, 0.999), weight_decay=0.01)

for epoch in range(100):  # loop over the dataset multiple times
    with torch.enable_grad():
        model_dis.train()
        model_de.train()
        model_rec.train()
        model_en.train()
        net1.train()
        net2.train()
        net3.train()
        running_loss, running_gan, running_coarse, running_recover = 0.0, 0.0, 0.0, 0.0
        running_cls1, running_cls2, running_cls3 = 0.0, 0.0, 0.0
        running_cls_wrong, running_cls_right = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer_net1.zero_grad()
            optimizer_net2.zero_grad()
            optimizer_net3.zero_grad()
            optimizer_en.zero_grad()
            optimizer_dis.zero_grad()
            optimizer_de.zero_grad()
            optimizer_rec.zero_grad()

            # forward + backward + optimize
            loss_cls1 = criterion(net1(inputs), labels)
            loss_cls1.backward()
            optimizer_net1.step()
            optimizer_net1.zero_grad()
            ##
            loss_cls2 = criterion(net2(inputs), labels)
            loss_cls2.backward()
            optimizer_net2.step()
            optimizer_net2.zero_grad()
            ##
            loss_cls3 = criterion(net3(inputs), labels)
            loss_cls3.backward()
            optimizer_net3.step()
            optimizer_net3.zero_grad()

            encoded = model_en(inputs)
            decoded = model_de(encoded)
            decoded_clamp = clamp_with_grad(decoded)
            recovered = model_rec(decoded)
            recovered_clamp = clamp_with_grad(recovered)

            gan_real = model_dis(inputs)
            gan_fake = model_dis(recovered.detach())
            gan_fake1 = model_dis(decoded.detach())
            loss_g_real = bce_with_logits_loss(gan_real,torch.ones_like(gan_real))
            loss_g_fake = bce_with_logits_loss(gan_fake,torch.zeros_like(gan_fake))
            loss_g_fake1 = bce_with_logits_loss(gan_fake1, torch.zeros_like(gan_fake))
            loss_gan = ((loss_g_fake+loss_g_fake1)/2+loss_g_real)/2
            loss_gan.backward()
            optimizer_dis.step()
            optimizer_dis.zero_grad()

            psnr_forward = psnr(postprocess(decoded_clamp), postprocess(inputs)).item()
            psnr_backward = psnr(postprocess(recovered_clamp), postprocess(inputs)).item()

            loss_cls_wrong = criterion(net1(decoded),labels) + \
                             criterion(net2(decoded),labels) + \
                             criterion(net3(decoded),labels)
            loss_cls_wrong /= 3
            loss_cls_right = criterion(net1(recovered),labels) + \
                             criterion(net2(recovered),labels) + \
                             criterion(net3(recovered),labels)
            loss_cls_right /= 3
            # loss_CW = cw_loss(should_be_right, labels)

            loss_recover = l1_loss(recovered,inputs)
            loss_coarse = l1_loss(decoded,inputs)
            gan_fake = model_dis(recovered)
            loss_g = bce_with_logits_loss(gan_fake,torch.ones_like(gan_fake))
            gan_fake1 = model_dis(decoded)
            loss_g1 = bce_with_logits_loss(gan_fake1, torch.ones_like(gan_fake))

            loss = loss_coarse+loss_recover
            loss += loss_g*0.01+loss_g1*0.01
            if epoch>=3 and psnr_forward>=psnr_thresh:
                loss += -cls_weight*loss_cls_wrong+2*cls_weight*loss_cls_right
            loss.backward()
            optimizer_en.step()
            optimizer_dis.step()
            optimizer_de.step()
            optimizer_rec.step()

            running_loss += loss.item()
            running_gan += loss_gan.item()
            running_coarse += psnr_forward
            running_cls1 += loss_cls1.item()
            running_cls2 += loss_cls2.item()
            running_cls3 += loss_cls3.item()
            running_recover += psnr_backward
            running_cls_right += loss_cls_right.item()
            running_cls_wrong += loss_cls_wrong.item()

            if i % print_step == print_step-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / print_step:.5f} '
                      f'gan: {running_gan / print_step:.5f} '
                      f'recover: {running_recover / print_step:.5f} '
                      f'vgg: {running_cls1 / print_step:.5f} '
                      f'res: {running_cls2 / print_step:.5f} '
                      f'goo: {running_cls3 / print_step:.5f} '
                      f'coarse: {running_coarse / print_step:.5f} '
                      f'cls_wrong: {running_cls_wrong / print_step:.5f} '
                      f'cls_right: {running_cls_right / print_step:.5f} ')
                running_loss, running_gan, running_coarse, running_recover = 0.0, 0.0, 0.0, 0.0
                running_cls1, running_cls2, running_cls3 = 0.0, 0.0, 0.0
                running_cls_wrong, running_cls_right = 0.0, 0.0
            if i % save_step == save_step-1:
                images = stitch_images(
                    postprocess(inputs),
                    postprocess(decoded),
                    postprocess(recovered),
                    img_per_row=1
                )

                name = f'./img/{str(epoch)}_{str(int(compression_rate*10))}.png'
                print('\nsaving sample ' + name)
                images.save(name)

        if epoch%10==9:
            print(f'saving model at epoch {epoch}')

            PATH = f'./vgg_{str(compression_rate*10)}.pth'
            torch.save(net1.state_dict(), PATH)
            PATH = f'./res_{str(compression_rate*10)}.pth'
            torch.save(net2.state_dict(), PATH)
            PATH = f'./google_{str(compression_rate*10)}.pth'
            torch.save(net3.state_dict(), PATH)
            PATH = f'./model_en_{str(compression_rate*10)}.pth'
            torch.save(model_en.state_dict(), PATH)
            PATH = f'./model_de_{str(compression_rate*10)}.pth'
            torch.save(model_de.state_dict(), PATH)
            PATH = f'./model_dis_{str(compression_rate*10)}.pth'
            torch.save(model_dis.state_dict(), PATH)
            ATH = f'./model_rec_{str(compression_rate * 10)}.pth'
            torch.save(model_rec.state_dict(), PATH)

    ## eval
    print(f'-------------- Start Evaluating Epoch {epoch} ------------------')
    running_loss, running_gan, running_coarse, running_recover = 0.0, 0.0, 0.0, 0.0
    running_cls1, running_cls2, running_cls3 = 0.0, 0.0, 0.0
    running_cls_wrong, running_cls_right = 0.0, 0.0
    with torch.no_grad():
        model_dis.eval()
        model_de.eval()
        model_rec.eval()
        model_en.eval()
        net1.eval()
        net2.eval()
        net3.eval()
        sum_batches = len(testloader)
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            encoded = model_en(inputs)
            decoded = model_de(encoded)
            decoded_clamp = clamp_with_grad(decoded)
            recovered = model_rec(decoded)
            recovered_clamp = clamp_with_grad(recovered)

            psnr_forward = psnr(postprocess(decoded_clamp), postprocess(inputs)).item()
            psnr_backward = psnr(postprocess(recovered_clamp), postprocess(inputs)).item()
            _, argmax = torch.max(net1(inputs), 1)
            running_cls1 += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net2(inputs), 1)
            running_cls2 += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net3(inputs), 1)
            running_cls3 += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net1(decoded), 1)
            running_cls_wrong += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net2(decoded), 1)
            running_cls_wrong += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net3(decoded), 1)
            running_cls_wrong += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net1(recovered), 1)
            running_cls_right += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net2(recovered), 1)
            running_cls_right += (labels == argmax.squeeze()).float().mean()
            _, argmax = torch.max(net3(recovered), 1)
            running_cls_right += (labels == argmax.squeeze()).float().mean()

            running_coarse += psnr_forward
            running_recover += psnr_backward

        running_cls_right /= 3
        running_cls_wrong /= 3
        print(f'[{epoch + 1}, {i + 1:5d}] '
              f'recover: {running_recover / sum_batches:.5f} '
              f'vgg: {running_cls1 / sum_batches:.5f} '
              f'res: {running_cls2 / sum_batches:.5f} '
              f'goo: {running_cls3 / sum_batches:.5f} '
              f'coarse: {running_coarse / sum_batches:.5f} '
              f'cls_wrong: {running_cls_wrong / sum_batches:.5f} '
              f'cls_right: {running_cls_right / sum_batches:.5f} ')

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))