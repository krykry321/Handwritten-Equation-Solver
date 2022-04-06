from models import shufflenet_v2_x0_5, shufflenet_v2_x2_0, mobilenet_v2
import torch
from tqdm import tqdm
import sys
from get_data import hand_dataset
from torch.utils.data import DataLoader as DataLoader

sys.setrecursionlimit(1000000)


def train_one_epoch(model, optimizer, data_loader, epoch, device='cuda'):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)
    for step, (img, label) in enumerate(data_loader):
        img, label = img.cuda(), label.cuda()
        pred = model(img)

        loss = loss_function(pred, label.squeeze())
        loss.backward()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        # 打印平均loss
        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, param_change=False, param=None, device='cuda'):
    if param_change:
        model.load_state_dict(torch.load(param), strict=False)
    model.cuda()
    model.eval()

    sum_num = torch.zeros(1).to(device)

    num_samples = len(data_loader.dataset)
    if param_change:
        print('\nValiadation dataset contains {0} items, using {1}'.format(num_samples, param))
    else:
        print('\nValiadation dataset contains {0} items'.format(num_samples))
    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...")

    # test_bar = tqdm(data_loader)
    for test_data in data_loader:
        test_imgs, test_labels = test_data
        test_imgs = test_imgs.cuda()
        test_labels = test_labels.cuda()
        test_labels = test_labels.squeeze(dim=1)  # [B,1] -> [B]
        outputs = model(test_imgs)
        predict_y = torch.max(outputs, dim=1)[1]
        sum_num += torch.eq(predict_y, test_labels).sum().item()
    test_acc = sum_num / num_samples
    print('Testing accuracy is', round(test_acc.item(), 3))
    return test_acc.item(), sum_num.item(), num_samples


if __name__ == '__main__':

    train_dir = ''
    save_dir = ''
    Epochs = 100
    Batch_size = 100
    Learning_rate = 0.000125

    # model = shufflenet_v2_x2_0(num_classes=14)
    model = mobilenet_v2(num_classes=14)
    model.cuda()
    dataset = hand_dataset(dir=train_dir, image_size=45)

    dataset = dataset

    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(Epochs):
        whole_len = len(dataset)
        split_len1 = int(0.5 * whole_len)
        split_len2 = whole_len - split_len1
        data1, data2 = torch.utils.data.random_split(dataset, [split_len1, split_len2])

        dataloader = DataLoader(data1, batch_size=Batch_size, shuffle=True, num_workers=8)
        testloader = DataLoader(data2, batch_size=Batch_size, shuffle=True, num_workers=8)

        mean_loss = train_one_epoch(model=model, data_loader=dataloader, optimizer=optimizer, epoch=epoch)

        acc, acc_num, img_num = evaluate(model=model,
                                         data_loader=testloader)
        print(f'第{epoch}个epoch的平均准确率为{acc}。\n')
        if acc > 0.9:
            torch.save(model.state_dict(),
                       f'{save_dir}/MobileNet_Epoch_{epoch} acc_{int(acc * 100)}.pth')
    pass
