from models import mobilenet_v2
from get_num import get_num_sym
import torch


def calculate(img):
    model = mobilenet_v2(num_classes=14)

    model.load_state_dict(torch.load(
        './1MobileNet_Epoch_64 acc_99 count1.pth'))
    imgs = get_num_sym(dir=img)

    model.cuda()
    model.eval()
    imgs = imgs.cuda()
    out = model(imgs)
    out = torch.softmax(out, dim=1)
    out = torch.argmax(out, dim=1)

    out = out.to("cpu").numpy()
    out = list(out)

    equation_string = ''

    for i in out:
        if i < 10:
            equation_string = equation_string + str(i)
        elif i == 10:
            equation_string = equation_string + '+'
        elif i == 11:
            equation_string = equation_string + '-'
        elif i == 12:
            equation_string = equation_string + '*'
        elif i == 13:
            equation_string = equation_string + '/'

    if out[0] > 9 or out[-1] > 9:
        return equation_string, 'Wrong equation or Wrong recognition'
    for i in range(len(out) - 2):
        if out[i] + out[i + 1] > 19:
            return equation_string, 'Wrong equation or Wrong recognition'

    # print(out)
    print('Equation: ', equation_string)
    print('Solution: ', eval(equation_string))
    return equation_string, eval(equation_string)


if __name__ == '__main__':
    img_dir = './23+54x768-90d12.jpg'
    calculate(img_dir)
    pass
