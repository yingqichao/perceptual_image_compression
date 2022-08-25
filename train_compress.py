import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from models.MIMOUNet import SimplePatchGAN, MIMOUNetv2_decoder, MIMOUNetv2_encoder, MIMOUNetv2
from metrics import PSNR, postprocess
from models.loss import CWLoss
from models.class_models.vgg import VGG
from models.class_models.resnet import ResNet18
from models.class_models.densenet import DenseNet121
from models.class_models.googlenet import GoogLeNet
from utils import stitch_images, clamp_with_grad
import torch.optim as optim
import argparse
from utils import diff_round

parser = argparse.ArgumentParser()
parser.add_argument('-checkpoint', type=int, default=0.0, help='')
parser.add_argument('-compression_rate', type=float, default=32, help='')
parser.add_argument('-cls_weight', type=float, default=0.005, help='')
parser.add_argument('-thresh', type=float, default=28.0, help='')
parser.add_argument('-batch', type=int, default=64, help='')
parser.add_argument('-epoch_thresh', type=int, default=3, help='')
parser.add_argument('-dataset', type=str, default='CIFAR10', help='')
parser.add_argument('-original_scale', type=int, default=32, help='')

args = parser.parse_args()
print(args)


########### Settings ################
original_scale = args.original_scale
print_step, save_step = 50, 750
num_epochs = 50
dataset = args.dataset
batch_size = args.batch
psnr_thresh = args.thresh # 28
cls_weight = args.cls_weight #0.005
compression_rate = args.compression_rate
print(f"scale is {compression_rate}")
#####################################
print(f"using {args.dataset}")
if "CIFAR10" in dataset:
    original_scale = 32
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    num_classes = 10
elif "CIFAR100" in dataset:
    original_scale = 32
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    num_classes = 100
elif "Caltech" in dataset:
    # original_scale = 224
    transform = transforms.Compose(
        [transforms.Resize((original_scale, original_scale)),
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.ImageFolder(
        '/groupshare/Caltech-101/train',
        transform,
    )
    testset = torchvision.datasets.ImageFolder(
        '/groupshare/Caltech-101/test',
        transform,
    )
    num_classes = 101
elif "CelebA" in dataset:
    # original_scale = 224
    transform = transforms.Compose(
        [transforms.Resize((original_scale, original_scale)),
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.ImageFolder(
        '/groupshare/CelebA-100/train',
        transform,
    )
    testset = torchvision.datasets.ImageFolder(
        '/groupshare/CelebA-100/test',
        transform,
    )
    num_classes = 100
elif "ImageNet" in dataset:
    # original_scale = 224
    transform = transforms.Compose(
        [transforms.Resize((original_scale, original_scale)),
            transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.ImageFolder(
        '/groupshare/MiniImageNet/train',
        transform,
    )
    testset = torchvision.datasets.ImageFolder(
        '/groupshare/MiniImageNet/test',
        transform,
    )
    num_classes = 1000
else:
    raise NotImplementedError("We only support CIFAR/MiniImageNet/Caltech/CelebA so far!!!")


trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########### Models (Total:8) #################
model_en = MIMOUNetv2_encoder(scale=compression_rate, original_scale=original_scale).cuda()
model_de = MIMOUNetv2_decoder(scale=compression_rate, original_scale=original_scale).cuda()
model_rec = MIMOUNetv2(scale=compression_rate,enemy=False, original_scale=original_scale).cuda()
model_enemy = MIMOUNetv2(scale=compression_rate,enemy=True, original_scale=original_scale).cuda()
# net = Simple_Class_Net().cuda()
model_dis = SimplePatchGAN().cuda()
net1 = torchvision.models.densenet121(pretrained=False)
net1.classifier = nn.Linear(1024,num_classes)
net1 = net1.cuda()
net2 = torchvision.models.resnet50(pretrained=False)
net2.fc = nn.Linear(2048,num_classes)
net2 = net2.cuda()
net3 = torchvision.models.vgg19_bn(pretrained=False)
net3.classifier = nn.Sequential(
    nn.Linear(in_features=25088, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=4096, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5, inplace=False),
    nn.Linear(in_features=4096, out_features=num_classes, bias=True),
)
net3 = net3.cuda()
########### Losses #################
quantization = diff_round
cw_loss = CWLoss(num_classes=num_classes).cuda()
psnr = PSNR(255.0).cuda()
criterion = nn.CrossEntropyLoss().cuda()
l1_loss = nn.SmoothL1Loss().cuda()
bce_with_logits_loss = nn.BCEWithLogitsLoss().cuda()


########### Checkpoints and Optimizers #################

if args.checkpoint!=0:
    PATH = f'./vgg_{str(int(compression_rate))}_{dataset}.pth'
    net1.load_state_dict(torch.load(PATH))
    PATH = f'./res_{str(int(compression_rate))}_{dataset}.pth'
    net2.load_state_dict(torch.load(PATH))
    PATH = f'./google_{str(int(compression_rate))}_{dataset}.pth'
    net3.load_state_dict(torch.load(PATH))
    PATH = f'./model_en_{str(int(compression_rate))}_{dataset}.pth'
    model_en.load_state_dict(torch.load(PATH))
    PATH = f'./model_de_{str(int(compression_rate))}_{dataset}.pth'
    model_de.load_state_dict(torch.load(PATH))
    PATH = f'./model_dis_{str(int(compression_rate))}_{dataset}.pth'
    model_dis.load_state_dict(torch.load(PATH))
    PATH = f'./model_rec_{str(int(compression_rate))}_{dataset}.pth'
    model_rec.load_state_dict(torch.load(PATH))
    PATH = f'./model_enemy_{str(int(compression_rate))}_{dataset}.pth'
    model_enemy.load_state_dict(torch.load(PATH))
    print("Models loaded.")

optimizer_dis = optim.AdamW(model_dis.parameters(),
                                 lr=2e-4)
optimizer_en = optim.AdamW(model_en.parameters(),
                                 lr=2e-4)
optimizer_de = optim.AdamW(model_de.parameters(),
                                 lr=2e-4)
optimizer_net1 = optim.AdamW(net1.parameters(),
                                 lr=2e-4)
optimizer_net2 = optim.AdamW(net2.parameters(),
                                 lr=2e-4)
optimizer_net3 = optim.AdamW(net3.parameters(),
                                 lr=2e-4)
optimizer_rec = optim.AdamW(model_rec.parameters(),
                                 lr=2e-4)
optimizer_enemy = optim.AdamW(model_enemy.parameters(),
                                 lr=2e-4)

########## train ###########################
for epoch in range(num_epochs):
    with torch.enable_grad():
        model_enemy.train()
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
        running_psnr_enemy, running_cls_enemy = 0.0, 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            # zero the parameter gradients
            optimizer_enemy.zero_grad()
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
            loss_cls2 = criterion(net2(inputs), labels)
            loss_cls2.backward()
            optimizer_net2.step()
            loss_cls3 = criterion(net3(inputs), labels)
            loss_cls3.backward()
            optimizer_net3.step()


            encoded = model_en(inputs)
            decoded = model_de(encoded)
            # decoded_clamp = clamp_with_grad(decoded)
            recovered = model_rec(decoded, encoded)
            # recovered_clamp = clamp_with_grad(recovered)

            ####### ENEMY #########
            if True: #epoch>=args.epoch_thresh:
                enemy_recovered = model_enemy(decoded.detach())
                loss_l1_enemy = l1_loss(enemy_recovered, inputs)

                loss_enemy = loss_l1_enemy
                if epoch>=args.epoch_thresh:
                    loss_cls_enemy = criterion(net1(enemy_recovered), labels) + \
                                     criterion(net2(enemy_recovered), labels) + \
                                     criterion(net3(enemy_recovered), labels)
                    loss_cls_enemy /= 3
                    loss_enemy += cls_weight*loss_cls_enemy
                    running_cls_enemy += loss_cls_enemy.item()

                loss_enemy.backward()
                optimizer_enemy.step()
                # optimizer_enemy.zero_grad()

                psnr_enemy = psnr(postprocess(enemy_recovered), postprocess(inputs)).item()
                running_psnr_enemy += psnr_enemy

            #######################

            # gan_real = model_dis(inputs)
            # gan_fake = model_dis(recovered.detach())
            # gan_fake1 = model_dis(decoded.detach())
            # loss_g_real = bce_with_logits_loss(gan_real,torch.ones_like(gan_real))
            # loss_g_fake = bce_with_logits_loss(gan_fake,torch.zeros_like(gan_fake))
            # loss_g_fake1 = bce_with_logits_loss(gan_fake1, torch.zeros_like(gan_fake))
            # loss_gan = ((loss_g_fake+loss_g_fake1)/2+loss_g_real)/2
            # loss_gan.backward()
            # optimizer_dis.step()
            # optimizer_dis.zero_grad()

            psnr_forward = psnr(postprocess(decoded), postprocess(inputs)).item()
            psnr_backward = psnr(postprocess(recovered), postprocess(inputs)).item()

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
            loss_coarse = l1_loss(decoded, inputs)

            # gan_fake = model_dis(recovered)
            # loss_g = bce_with_logits_loss(gan_fake,torch.ones_like(gan_fake))
            # gan_fake1 = model_dis(decoded)
            # loss_g1 = bce_with_logits_loss(gan_fake1, torch.ones_like(gan_fake))

            loss = loss_coarse+loss_recover
            # loss += loss_g*0.01+loss_g1*0.01
            if epoch>=args.epoch_thresh and psnr_forward>=psnr_thresh:
                ####### enemy step ########
                enemy_recovered = model_enemy(decoded)
                loss_cls_enemy1 = criterion(net1(enemy_recovered), labels) + \
                                 criterion(net2(enemy_recovered), labels) + \
                                 criterion(net3(enemy_recovered), labels)
                loss_cls_enemy1 /= 3

                loss += -4*cls_weight*loss_cls_enemy1
                loss += -cls_weight*loss_cls_wrong #+1*cls_weight*loss_cls_right
            loss.backward()
            optimizer_en.step()
            optimizer_dis.step()
            optimizer_de.step()
            optimizer_rec.step()

            running_loss += loss.item()
            # running_gan += loss_gan.item()
            running_coarse += psnr_forward
            running_cls1 += loss_cls1.item()
            running_cls2 += loss_cls2.item()
            running_cls3 += loss_cls3.item()
            running_recover += psnr_backward
            running_cls_right += loss_cls_right.item()
            running_cls_wrong += loss_cls_wrong.item()

            if i % print_step == print_step-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1}, {compression_rate}] loss: {running_loss / print_step:.5f} '
                      f'gan: {running_gan / print_step:.5f} '
                      f'recover: {running_recover / print_step:.5f} '
                      f'vgg: {running_cls1 / print_step:.5f} '
                      f'res: {running_cls2 / print_step:.5f} '
                      f'goo: {running_cls3 / print_step:.5f} '
                      f'coarse: {running_coarse / print_step:.5f} '
                      f'cls_wrong: {running_cls_wrong / print_step:.5f} '
                      f'cls_right: {running_cls_right / print_step:.5f} '
                      f'enemy_psnr: {running_psnr_enemy / print_step:.5f} '
                      f'enemy_cls: {running_cls_enemy / print_step:.5f} '
                      )
                running_loss, running_gan, running_coarse, running_recover = 0.0, 0.0, 0.0, 0.0
                running_cls1, running_cls2, running_cls3 = 0.0, 0.0, 0.0
                running_cls_wrong, running_cls_right = 0.0, 0.0
                running_psnr_enemy, running_cls_enemy = 0.0, 0.0

            ######## save images ################
            if i % save_step == save_step-1:
                images = stitch_images(
                    postprocess(inputs),
                    postprocess(decoded),
                    postprocess(recovered),
                    img_per_row=1
                )

                name = f'./img/{str(epoch)}_{str(int(compression_rate*10))}_{dataset}.png'
                print('\nsaving sample ' + name)
                images.save(name)

        ########## save checkpoints #############
        if epoch%5==4:
            print(f'saving model at epoch {epoch}')

            PATH = f'./vgg_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(net1.state_dict(), PATH)
            PATH = f'./res_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(net2.state_dict(), PATH)
            PATH = f'./google_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(net3.state_dict(), PATH)
            PATH = f'./model_en_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(model_en.state_dict(), PATH)
            PATH = f'./model_de_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(model_de.state_dict(), PATH)
            PATH = f'./model_dis_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(model_dis.state_dict(), PATH)
            PATH = f'./model_rec_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(model_rec.state_dict(), PATH)
            PATH = f'./model_enemy_{str(int(compression_rate))}_{dataset}.pth'
            torch.save(model_enemy.state_dict(), PATH)

    ############ eval ##############################
    print(f'-------------- Start Evaluating Epoch {epoch} ------------------')
    running_loss, running_gan, running_coarse, running_recover = 0.0, 0.0, 0.0, 0.0
    running_cls1, running_cls2, running_cls3 = 0.0, 0.0, 0.0
    running_cls_wrong, running_cls_right = 0.0, 0.0
    running_psnr_enemy, running_cls_enemy = 0.0, 0.0
    with torch.no_grad():
        model_enemy.eval()
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
            encoded = quantization(encoded)
            decoded = model_de(encoded)
            # decoded_clamp = clamp_with_grad(decoded)
            recovered = model_rec(decoded, encoded)
            enemy_recovered = model_enemy(decoded)
            # recovered_clamp = clamp_with_grad(recovered)

            psnr_forward = psnr(postprocess(decoded), postprocess(inputs)).item()
            psnr_backward = psnr(postprocess(recovered), postprocess(inputs)).item()
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
            _, argmax = torch.max(net1(enemy_recovered), 1)
            if epoch>=args.epoch_thresh:
                psnr_enemy = psnr(postprocess(enemy_recovered), postprocess(inputs)).item()
                running_psnr_enemy += psnr_enemy
                running_cls_enemy += (labels == argmax.squeeze()).float().mean()
                _, argmax = torch.max(net2(enemy_recovered), 1)
                running_cls_enemy += (labels == argmax.squeeze()).float().mean()
                _, argmax = torch.max(net3(enemy_recovered), 1)
                running_cls_enemy += (labels == argmax.squeeze()).float().mean()

            running_coarse += psnr_forward
            running_recover += psnr_backward

        running_cls_enemy /= 3
        running_cls_right /= 3
        running_cls_wrong /= 3
        print(f'[{epoch + 1}, {i + 1:5d}] '
              f'recover: {running_recover / sum_batches:.5f} '
              f'vgg: {running_cls1 / sum_batches:.5f} '
              f'res: {running_cls2 / sum_batches:.5f} '
              f'goo: {running_cls3 / sum_batches:.5f} '
              f'coarse: {running_coarse / sum_batches:.5f} '
              f'cls_wrong: {running_cls_wrong / sum_batches:.5f} '
              f'cls_right: {running_cls_right / sum_batches:.5f} '
              f'cls_enemy: {running_cls_enemy / sum_batches:.5f} '
              )

# dataiter = iter(testloader)
# images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))