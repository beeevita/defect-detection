import pandas as pd                 #导入pandas
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import optimize
import random
#train_loss,train_acc,test_loss,test_acc

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def loadtxt(name):
    data = {"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[]}
    with open(name,"r+",encoding="utf-8") as f:
        fdata = f.readlines()
        #print(fdata)
        fdata=fdata[1:]
        for item in fdata:
            item=item.strip()
            item=item.split(",")
            print(item)
            if item==[]:
                break
            data["train_loss"].append(float(item[0]))
            data["train_acc"].append(float(item[1]))
            data["test_loss"].append(float(item[2]))
            data["test_acc"].append(float(item[3]))
        #print(data)
    print(name+"  has load.")
    return data
# #20类
# resnet = loadtxt("resnet.txt")
# res2net = loadtxt("res2net.txt")
res2net_bi = loadtxt("res2net_bi.txt")
res2net_mish = loadtxt("res2net_mish.txt")
res2net_conv7to3= loadtxt("res2net_conv7to3.txt")
#
#
# def findmax(list):
#     max = 0
#     index = 0
#     for i,item in enumerate(list):
#         if item>max:
#             max=item
#             index=i
#     return index,str(max)[:5]
#
# def findmin(list):
#     max = 100000
#     index = 0
#     for i,item in enumerate(list):
#         if item<max:
#             max=item
#             index=i
#     return index,str(max)[:5]
#
# def displaysingle(data,title):
#     epoch_num = len(data["train_loss"])
#     epoch_list = [i+1 for i in range(epoch_num)]
#     p = int(-epoch_num/8)
#     train_acc_mindex, train_acc_max = findmax(data["train_acc"])
#     #plt.text(train_acc_mindex,float(train_loss_max),train_loss_max)
#
#     test_acc_mindex, test_acc_max = findmax(data["test_acc"])
#     #plt.text(train_acc_mindex, float(train_loss_max), train_loss_max)
#
#     plt.plot(epoch_list, data["train_loss"], c="red",label="train_loss")
#     plt.text(epoch_list[p],data["train_loss"][-1],"train_loss")
#     plt.plot(epoch_list, data["train_acc"], marker='+', mfc='r', c="red",label="train_acc")
#     plt.text(epoch_list[p], data["train_acc"][-1], "train_acc max:"+train_acc_max)
#
#     plt.plot(epoch_list, data["test_loss"],  c="blue",label="test_loss")
#     plt.text(epoch_list[p], data["test_loss"][-1], "test_loss")
#     plt.plot(epoch_list, data["test_acc"], marker='+', mfc='blue', c="blue",label="test_acc")
#     plt.text(epoch_list[p], data["test_acc"][-1], "test_acc max:"+test_acc_max)
#
#     plt.legend(loc=0, ncol=1)
#     plt.title(title)
#     plt.xlabel('epoches')
#
#     plt.ylabel('acc/loss')
#     plt.savefig("result/"+title+".jpg")
#     plt.show()

# displaysingle(res2net_conv7to3,"res2net_conv7to3")
# displaysingle(res2net,"res2net")
# displaysingle(resnet,"resnet")
# displaysingle(res2net_mish,"res2net_mish")
# displaysingle(res2net_bi,"res2net_bi")


def compareall(kind,num,c):

    k=0
    epoch_list = [i + 1 for i in range(50)]
    plt.plot(epoch_list, res2net_conv7to3[kind], c=sns.xkcd_rgb[c[k]],label="res2net_conv7to3")
    plt.text(epoch_list[-5], res2net_conv7to3[kind][-1], "res2net_conv7to3")
    k+=1
    # plt.plot(epoch_list,res2net[kind], c=sns.xkcd_rgb[c[k]],label="res2net")
    # plt.text(epoch_list[-5], res2net[kind][-1], "res2net")
    # k += 1
    # plt.plot(epoch_list, resnet[kind], c=sns.xkcd_rgb[c[k]],label="resnet")
    # plt.text(epoch_list[-5], resnet[kind][-1], "resnet")
    # k += 1
    plt.plot(epoch_list, res2net_mish[kind], c=sns.xkcd_rgb[c[k]],label="res2net_mish")
    plt.text(epoch_list[-5], res2net_mish[kind][-1], "res2net_mish")
    k += 1
    plt.plot(epoch_list, res2net_bi[kind], c=sns.xkcd_rgb[c[k]],label="res2net_bi")
    plt.text(epoch_list[-5], res2net_bi[kind][-1], "res2net_bi")
    k += 1
    # plt.plot(epoch_list, se_v3_20c[kind], c=sns.xkcd_rgb[c[k]])
    # plt.text(epoch_list[-5], se_v3_20c[kind][-1], "se_v3_20c")
    # k += 1
    # plt.plot(epoch_list, se_origin20c[kind], c=sns.xkcd_rgb[c[k]])
    # plt.text(epoch_list[-5], se_origin20c[kind][-1], "se_origin20c")
    # k += 1
    plt.legend(loc=0, ncol=1)

    plt.title(kind)
    plt.xlabel('epoches')

    plt.ylabel('acc/loss')
    plt.savefig("result/" + kind + ".jpg")
    plt.show()

num = 5
c=[]
for i in range(num):
   c.append(random.choice(list(sns.xkcd_rgb)))
compareall("train_loss",num,c)
compareall("train_acc",num,c)
compareall("test_loss",num,c)
compareall("test_acc",num,c)

#color = random.choice(list(sns.xkcd_rgb))
#
#     plt.plot(x1, y1, c=sns.xkcd_rgb[color])

#
# data_path = "data_NoNAN.xlsx"
# data = pd.read_excel(data_path)
# data = data.drop("Unnamed: 0",axis=1)
# print(data)
# data = data.values.tolist()
# print(data)
# m = len(data[0])
# x = [i for i in range(1,len(data)+1)]
# k = []
# print(x)
# label =['A','B','C','D','E',
#                      'F','G','H','I','J',
#                      'K','L','M','N']
# for i in range(m):
#     team_list=[]
#     for j in range(len(data)):
#        team_list.append(data[j][i])
#     #print(team_list)
#     A1, B1 = optimize.curve_fit(f_1, x, team_list)[0]
#     x1 = np.arange(1, len(data)+1, 0.01)  # 30和75要对应x0的两个端点，0.01为步长
#     y1 = A1 * x1 + B1
#     k.append(A1*100)
#     color = random.choice(list(sns.xkcd_rgb))
#
#     plt.plot(x1, y1, c=sns.xkcd_rgb[color])
#     plt.text(x1[-1],y1[-1],label[i])
#
#     print(i,A1,B1)
#
# plt.title("上升潜力模型")
# plt.xlabel('t')
#
# plt.ylabel('原始数据分数')
# plt.savefig("上升潜力模型_1.jpg")
# plt.show()
