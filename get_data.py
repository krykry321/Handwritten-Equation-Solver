import os
import torch.utils.data as data
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2 as cv


class hand_dataset(data.Dataset):
    def __init__(self, dir, transform_change=False, transform=None, image_size=224):
        self.list_img = []
        self.list_label = []
        self.data_size = 0
        self.img_size = image_size
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.CenterCrop((self.img_size, self.img_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_change = transform_change
        if self.transform_change:
            self.transform = transform
        b = os.listdir(dir)[0]
        if os.path.isdir(dir + '/' + b):
            for folder in os.listdir(dir):
                for images in os.listdir(dir + '/' + folder):
                    self.list_img.append(dir + '/' + folder + '/' + images)
                    folder_index = int(folder)
                    self.data_size += 1
                    self.list_label.append(folder_index)
        else:
            count = -1
            for i in dir[::-1]:
                count += 1
                if i == '/':
                    break
            c = dir[len(dir) - count:]
            for images in os.listdir(dir):
                self.list_img.append(dir + '/' + images)
                folder_index = int(c)
                self.data_size += 1
                self.list_label.append(folder_index)

    def __getitem__(self, item):
        img = Image.open(self.list_img[item])
        label = self.list_label[item]
        return self.transform(img), torch.LongTensor([label])

    def __len__(self):
        return self.data_size


if __name__ == '__main__':

    train_data = hand_dataset(dir='./nums and symbols')

    for i in train_data:
        if i[1].item() == 2:
            img = i[0]
            label = i[1].item()
            break

    img = torch.squeeze(img, dim=0)
    print(img.size())

    img = img.numpy()
    print(img.shape)
    print(img)
    cv.imshow('{}'.format(label), img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    pass
