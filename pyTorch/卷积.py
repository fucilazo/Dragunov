import cv2
import torch as t
from torch import nn
from torch.autograd import Variable as V
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

to_tensor = ToTensor()
to_pil = ToPILImage()
# img = Image.open('city.png')
# img = img.convert('L')
img = cv2.imread('test.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 输入是一个batch，batch_size=1
img_tensor = to_tensor(img_gray).unsqueeze(0)

# 锐化卷积核
kernel = t.ones(3, 3)/-9.
kernel[1][1] = 1
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)

out = conv(V(img_tensor))
out_img = to_pil(out.data.squeeze(0))
out_img.show()

