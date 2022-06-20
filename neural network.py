from pandas import Series as se
from pandas import DataFrame as df
from scipy.io import loadmat
import scipy.stats
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import SGD

from matplotlib import pyplot as plt

import random

import sys

from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.utils import get_custom_objects

from scipy import stats
########################################################
# Folder Path
########################################################
folder_dir = 'E:/关于凋落物模拟/LIDET/LIRET数据/再处理/outmean/MCMC6_paramMean/机器学习/'
output_folder = 'E:/关于凋落物模拟/LIDET/LIRET数据/再处理/outmean/MCMC6_paramMean/机器学习/'
csdata = 'paramBest'
# csdata = 'paramMean'

########################################################
# Environment Setting
########################################################
# 这就是一个用来实现深度学习算法的开源框架，基本数据结构是Tensor，然后Flow指的是dataflow，也就是数据流。
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
sess = tf.compat.v1.Session(config=config)

# 损失函数
# 损失函数用于量化从站点级数据同化中获得的预测值与实际参数值之间的差异。损失函数值越低，预测值与实际参数值越接近，模型性能越好。
# 损失函数可以是均方误差（在Keras中表示为“均方误差”）或其他可以量化预测值和目标值之间差异的函数损失函数可以是均方误差（在Keras中表示为“均方误差”）
# 或其他可以量化预测值和目标值之间差异的函数
nn_loss = 'mse'  # mse joint_loss

# 优化器
# 在神经网络中，我们利用优化器根据损失函数的结果来调整神经元的权值。
# 在理想情况下，经过足够的训练后，优化器最终会引导神经网络到达一个点，在这个点上，损失函数保留了它可以追求的最小值。
# 此时，我们认为神经网络达到了全局最优。因此，神经网络的预测将最接近训练目标
# Keras预定义的其他激活函数或您自己定义的函数。Keras为优化器提供了几个选项。选择“Adam”、“Adadelta”或“RMSprop”通常是一个好的开始
nn_optimizer = 'Adam'

# 批量大小
# 批处理大小定义了在每个epoch中处理整个训练数据集时希望作为批处理的训练数据数量。
# 例如，对于一个有10000个样本的训练集，如果我们将批次大小设置为50，则将需要200次迭代
# 将批大小设置为16、32 或64是一个合理的开始
nn_batch_size = 128

# epoch number决定了深度学习算法经过整个数据集进行训练的次数。
# 在每个阶段，神经网络可以根据损失函数的结果提出一组神经元权值，并通过优化器对其进行调整。
# 在不同的深度学习应用程序中，epoch的数量从数百到数千不等。
# 你可以尝试不同的数来找到使损失函数值最小化的最佳历元数，这样神经网络就可以准确地预测训练目标。
nn_epochs = 4800

# 每层神经元数 四个数就是4层
# 神经元是神经网络的基本单位。
# 它们分布在神经网络的每一隐含层(图2)，接收来自输入层或前一层的信息，并生成下一层所有可能的预测或作为最终输出。
# 隐层数决定了神经网络的深度。每个隐含层的神经元数量同时控制着神经网络的宽度。隐层数和神经元数的选择在很大程度上是经验的。
nn_layer_num = [256, 512, 512, 256]

# 每层的下降率
# 如果我们不想让一小群神经元的表现在训练后的神经网络的最终预测中有太大的影响，Dropout提供了一个进一步的选择。
# “退出”选项允许我们在每个优化阶段随机排列某些特定百分比的神经元。然后训练神经网络在预测时不太依赖任何特定的神经元，从而提高其鲁棒性。
# 如果你检查练习1中的默认设置，你会发现我们在神经网络训练中使用了dropout选项。
nn_drop_ratio = [0.3, 0.5, 0.5, 0.3]

# 自定义激活函数
# 除了神经网络的四个基本元素外，我们使用激活函数来生成神经元权值。我们通常使用非线性激活函数，使神经网络能够探索输入和最终输出之间的非线性。
# 激活函数可以是校正线性单元(在Keras中表示为ReLU)，双曲正切函数(在Keras中表示为tanh)，
# sigmoid函数(在Keras中表示为sigmoid)，其他由Keras预先定义的激活函数，或您自己定义的函数。
use_custom_activation = 0
nn_activation = [None] * len(nn_layer_num) #激活函数可以是校正的线性单位

if use_custom_activation == 1:
    # define activation function 定义激活函数
    def custom_activation(x):
        custom_activation = tf.keras.activations.relu(x, alpha=0.1)
        return custom_activation


    for ilayer in range(len(nn_layer_num)):
        get_custom_objects().update({'custom_activation_' + str(ilayer): Activation(custom_activation)})
        nn_activation[ilayer] = 'custom_activation_' + str(ilayer)
else:
    nn_activation = ['relu', 'relu', 'relu', 'relu']


########################################################
# Import para info after bayesian method 使用贝叶斯方法导入para信息
########################################################
# laod para info after MCMC MCMC后加载para信息   6是去掉种类和类型   5是去掉种类
para_without_test = loadmat(folder_dir + 'input/'+csdata+'_EnvInfo去掉凋落物种类_test.mat')
para_without_test = df(para_without_test[csdata+'_EnvInfo_test'])
param_test = para_without_test.iloc[:, 0:5]
para_without_trans = loadmat(folder_dir + 'input/'+csdata+'_EnvInfo去掉凋落物种类_train.mat')
para_without_trans = df(para_without_trans[csdata+'_EnvInfo_train'])
param_trans = para_without_trans.iloc[:, 0:5]
para_names = ['Vmax', 'Km', 'cue', 'kb', 'kl']
# param.columns = para_names  #给每列加上参数名
# 环境信息
# env_info = loadmat(folder_dir + 'input_data/EnvInfo4NN_SoilGrids.mat')
env_info_trans = para_without_trans.iloc[:, 5:]
env_info_test = para_without_test.iloc[:, 5:]
env_info_names = ['TYPE', 'Climate', 'GLOBCOVER', 'PH_0_20cm', 'TAWC_0_20C', 'ELEV', 'PDSI1980_2',
                  'AET1980_20',	'PET1980_20', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7',
                  'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16',
                  'BIO17', 'BIO18', 'BIO19']

# env_info_test.columns = env_info_names

# environmental info of global grids 全球网格的环境信息
grid_env_info = loadmat(folder_dir + 'input/合成Global_data.mat')
grid_env_info = df(grid_env_info['Global_data'])
grid_env_info_names = ['Lon', 'Lat', 'Climate', 'GLOBCOVER', 'PH_0_20cm', 'TAWC_0_20C', 'ELEV', 'PDSI1980_2',
                       'AET1980_20', 'PET1980_20', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7',
                       'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16',
                       'BIO17', 'BIO18', 'BIO19', 'mosaic']
grid_env_info_names2 = ['Climate', 'GLOBCOVER', 'PH_0_20cm', 'TAWC_0_20C', 'ELEV', 'PDSI1980_2',
                        'AET1980_20', 'PET1980_20', 'BIO1', 'BIO2', 'BIO3', 'BIO4', 'BIO5', 'BIO6', 'BIO7',
                        'BIO8', 'BIO9', 'BIO10', 'BIO11', 'BIO12', 'BIO13', 'BIO14', 'BIO15', 'BIO16',
                        'BIO17', 'BIO18', 'BIO19']

grid_env_info.columns = grid_env_info_names

# variables used in training the NN 用于训练神经网络的变量
var4nn = env_info_names

########################################################
# specify the profiles to US continent 指定大陆的概况
########################################################
# profiles that in use 正在使用的配置文件 取出所有有值的行
# # delete veired data 删除审核数据
current_grid = np.array(grid_env_info.loc[(grid_env_info.mosaic == 1), grid_env_info_names2])
youzhi = np.array(grid_env_info.mosaic)
valid_grid_loc = np.array(range(len(youzhi))) + 1
valid_grid_loc = valid_grid_loc[youzhi == 1]

# 添加凋落物种类的列
ye = np.ones(current_grid.shape[0])
gen = np.ones(current_grid.shape[0])*2
wa = np.ones(current_grid.shape[0])*3
wb = np.ones(current_grid.shape[0])*4

ye_current_grid = np.insert(current_grid, 0, values=ye, axis=1)
gen_current_grid = np.insert(current_grid, 0, values=gen, axis=1)
wa_current_grid = np.insert(current_grid, 0, values=wa, axis=1)
wb_current_grid = np.insert(current_grid, 0, values=wb, axis=1)
# 站点对应的环境数据
env_info_trans = np.array(env_info_trans)
env_info_test = np.array(env_info_test)
transdata_x = np.zeros((env_info_trans.shape[0], env_info_trans.shape[1]), dtype=float)
testdata_x = np.zeros((env_info_test.shape[0], env_info_test.shape[1]), dtype=float)

# mcmc估计出的参数
param_trans = np.array(param_trans)
param_test = np.array(param_test)
transdata_y = np.zeros((param_trans.shape[0], param_trans.shape[1]), dtype=float)
testdata_y = np.zeros((param_test.shape[0], param_test.shape[1]), dtype=float)

# normalization for x 归一化x
# 针对网格数据归一化，我用的是测量数据
transdata_x[:, 0] = (env_info_trans[:, 0] - 1) / (4 - 1)
testdata_x[:, 0] = (env_info_test[:, 0] - 1) / (4 - 1)
ye_current_grid[:, 0] = (ye_current_grid[:, 0] - 1) / (4 - 1)
gen_current_grid[:, 0] = (gen_current_grid[:, 0] - 1) / (4 - 1)
wa_current_grid[:, 0] = (wa_current_grid[:, 0] - 1) / (4 - 1)
wb_current_grid[:, 0] = (wb_current_grid[:, 0] - 1) / (4 - 1)
for ivar in range(len(grid_env_info_names2)):
    transdata_x[:, ivar+1] = (env_info_trans[:, ivar+1] - min(current_grid[:, ivar])) / (
            max(current_grid[:, ivar]) - min(current_grid[:, ivar]))
    testdata_x[:, ivar+1] = (env_info_test[:, ivar + 1] - min(current_grid[:, ivar])) / (
            max(current_grid[:, ivar]) - min(current_grid[:, ivar]))
    ye_current_grid[:, ivar+1] = (ye_current_grid[:, ivar+1] - min(current_grid[:, ivar])) / (
            max(current_grid[:, ivar]) - min(current_grid[:, ivar]))
    gen_current_grid[:, ivar + 1] = (gen_current_grid[:, ivar + 1] - min(current_grid[:, ivar])) / (
            max(current_grid[:, ivar]) - min(current_grid[:, ivar]))
    wa_current_grid[:, ivar + 1] = (wa_current_grid[:, ivar + 1] - min(current_grid[:, ivar])) / (
            max(current_grid[:, ivar]) - min(current_grid[:, ivar]))
    wb_current_grid[:, ivar + 1] = (wb_current_grid[:, ivar + 1] - min(current_grid[:, ivar])) / (
            max(current_grid[:, ivar]) - min(current_grid[:, ivar]))

    # transdata_x[:, ivar] = (env_info_trans[:, ivar] - min(min(env_info_trans[:, ivar]), min(env_info_test[:, ivar]))) / (
    #         max(max(env_info_trans[:, ivar]), max(env_info_test[:, ivar])) - min(min(env_info_trans[:, ivar]), min(env_info_test[:, ivar])))
    # testdata_x[:, ivar] = (env_info_test[:, ivar] - min(min(env_info_trans[:, ivar]), min(env_info_test[:, ivar]))) / (
    #         max(max(env_info_trans[:, ivar]), max(env_info_test[:, ivar])) - min(min(env_info_trans[:, ivar]), min(env_info_test[:, ivar])))
    # current_grid[:, ivar] = (current_grid[:, ivar] - min(current_grid[:, ivar])) / (
    #             max(current_grid[:, ivar]) - min(current_grid[:, ivar]))


minmaxdata_y = np.zeros((2, len(para_names)), dtype=float)
for ivar in range(len(para_names)):
    transdata_y[:, ivar] = (param_trans[:, ivar] - min(min(param_trans[:, ivar]), min(param_test[:, ivar]))) / (
            max(max(param_trans[:, ivar]), max(param_test[:, ivar])) - min(min(param_trans[:, ivar]), min(param_test[:, ivar])))
    testdata_y[:, ivar] = (param_test[:, ivar] - min(min(param_trans[:, ivar]), min(param_test[:, ivar]))) / (
            max(max(param_trans[:, ivar]), max(param_test[:, ivar])) - min(min(param_trans[:, ivar]), min(param_test[:, ivar])))
    print('mindata_y', ivar, ':', min(min(param_trans[:, ivar]), min(param_test[:, ivar])))
    print('maxndata_y', ivar, ':', max(max(param_trans[:, ivar]), max(param_test[:, ivar])))
    minmaxdata_y[0, ivar] = min(min(param_trans[:, ivar]), min(param_test[:, ivar]))
    minmaxdata_y[1, ivar] = max(max(param_trans[:, ivar]), max(param_test[:, ivar]))


########################################################
# Build NN 构建神经网络
########################################################
# split into input and outputs 分成输入和输出
train_x = transdata_x
train_y = transdata_y

test_x = testdata_x
test_y = testdata_y

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# define a joint loss 确定共同损失
para_mse = 1000 * 0.5
para_ratio = 10 * 0.5


def joint_loss(y_true, y_pred):
    # mse
    mse_loss = K.mean(K.square(y_true - y_pred))
    # mean absolute ratio  error 平均绝对比误差
    ratio_loss = K.mean(K.abs((y_true - y_pred) / y_true))
    # return the joint loss 退回共同损失
    return para_mse * mse_loss + para_ratio * ratio_loss


def ratio_loss(y_true, y_pred):
    # mean absolute ratio  error 平均绝对比误差
    ratio_loss = K.mean(K.abs((y_true - y_pred) / y_true))
    return ratio_loss


# design network 设计网络
model = Sequential()

# hidden layers 隐含层数
for ilayer in range(len(nn_layer_num)):
    if use_custom_activation == 1:
        if ilayer == 0:
            model.add(Dense(nn_layer_num[ilayer], input_dim=len(var4nn)))
            model.add(Activation(custom_activation, name=nn_activation[ilayer]))
            model.add(Dropout(nn_drop_ratio[ilayer]))
        else:
            model.add(Dense(nn_layer_num[ilayer]))
            model.add(Activation(custom_activation, name=nn_activation[ilayer]))
            model.add(Dropout(nn_drop_ratio[ilayer]))
    else:
        if ilayer == 0:
            model.add(Dense(nn_layer_num[ilayer], input_dim=len(var4nn), activation=nn_activation[ilayer]))
            model.add(Dropout(nn_drop_ratio[ilayer]))
        else:
            model.add(Dense(nn_layer_num[ilayer], activation=nn_activation[ilayer]))
            model.add(Dropout(nn_drop_ratio[ilayer]))

model.add(Dense(len(para_names)))

# define loss and optimizer 定义损失和优化器
if nn_loss == 'joint_loss':
    model.compile(loss=joint_loss, optimizer=nn_optimizer, metrics=['accuracy'])
else:
    model.compile(loss=nn_loss, optimizer=nn_optimizer, metrics=['accuracy'])

model.summary()

# fit network 适合网络
history = model.fit(x=train_x, y=train_y, epochs=nn_epochs, batch_size=nn_batch_size, validation_split=0.2)
# save the trained model 保存经过训练的模型
model.save(output_folder + 'output全球/'+csdata+'_nn_model.h5')

# predict para based on the trained NN 基于训练的神经网络预测参数
nn_predict = model.predict(test_x)
ye_grid_predict = model.predict(ye_current_grid)
gen_grid_predict = model.predict(gen_current_grid)
wa_grid_predict = model.predict(wa_current_grid)
wb_grid_predict = model.predict(wb_current_grid)

########################################################
# Visualization of NN Results 神经网络结果可视化
########################################################
# loss function 损失函数
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.yscale('log')
plt.xscale('log')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(output_folder + 'output全球/'+csdata+'_loss_nn.png')
plt.close()

# predicted paras 预测值
xylim = np.array([[0,100],
                 [500,5000],
                 [0,0.3],
                 [0,0.015],
                 [0,0.005]])

for ipara in range(len(para_names)):
    # 相关系数和P值
    r, p = stats.pearsonr(test_y[:, ipara]*(minmaxdata_y[1, ipara]-minmaxdata_y[0, ipara])+minmaxdata_y[0, ipara],
                          nn_predict[:, ipara]*(minmaxdata_y[1, ipara]-minmaxdata_y[0, ipara])+minmaxdata_y[0, ipara])

    ax = plt.subplot(3, 2, ipara + 1)
    ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--', c='k', label="1:1 line")
    plt.xlim(xylim[ipara])
    plt.ylim(xylim[ipara])
    plt.plot(test_y[:, ipara]*(minmaxdata_y[1, ipara]-minmaxdata_y[0, ipara])+minmaxdata_y[0, ipara],
             nn_predict[:, ipara]*(minmaxdata_y[1, ipara]-minmaxdata_y[0, ipara])+minmaxdata_y[0, ipara],
             'o', markersize=10, markeredgecolor='red', markerfacecolor='red')
    plt.title(para_names[ipara])
    print(para_names[ipara], r, p)
    # ax.text(30, 20, 'correlation = ')
    # ax.text(32, 20, r)
    # ax.text(30, 22, 'P = ')
    # ax.text(32, 22, p)


plt.savefig(output_folder + 'output全球/'+csdata+'_para_nn.png')
plt.close()

########################################################
# Save the output
########################################################
# nn_site_loc = valid_loc[test_loc]
for ipara in range(len(para_names)):
    test_y[:, ipara] = test_y[:, ipara] * (minmaxdata_y[1, ipara] - minmaxdata_y[0, ipara]) + minmaxdata_y[0, ipara]
    nn_predict[:, ipara] = nn_predict[:, ipara] * (minmaxdata_y[1, ipara] - minmaxdata_y[0, ipara]) + minmaxdata_y[0, ipara]
    ye_grid_predict[:, ipara] = ye_grid_predict[:, ipara] * (minmaxdata_y[1, ipara] - minmaxdata_y[0, ipara]) + minmaxdata_y[0, ipara]
    gen_grid_predict[:, ipara] = gen_grid_predict[:, ipara] * (minmaxdata_y[1, ipara] - minmaxdata_y[0, ipara]) + minmaxdata_y[0, ipara]
    wa_grid_predict[:, ipara] = wa_grid_predict[:, ipara] * (minmaxdata_y[1, ipara] - minmaxdata_y[0, ipara]) + minmaxdata_y[0, ipara]
    wb_grid_predict[:, ipara] = wb_grid_predict[:, ipara] * (minmaxdata_y[1, ipara] - minmaxdata_y[0, ipara]) + minmaxdata_y[0, ipara]


# 相关系数矩阵
# corr_para = [None] * len(para_names)
# corr_para = np.corrcoef(test_y,nn_predict)[0, 1]
# print(corr_para)

np.savetxt(output_folder + 'output全球/'+csdata+'_nn_测试参数.csv', test_y, delimiter=',')
np.savetxt(output_folder + 'output全球/'+csdata+'_nn_预测参数.csv', nn_predict, delimiter=',')
np.savetxt(output_folder + 'output全球/'+csdata+'_grid_预测全球叶参数.csv', ye_grid_predict, delimiter=',')
np.savetxt(output_folder + 'output全球/'+csdata+'_grid_预测全球根参数.csv', gen_grid_predict, delimiter=',')
np.savetxt(output_folder + 'output全球/'+csdata+'_grid_预测全球wa参数.csv', wa_grid_predict, delimiter=',')
np.savetxt(output_folder + 'output全球/'+csdata+'_grid_预测全球wb参数.csv', wb_grid_predict, delimiter=',')
np.savetxt(output_folder + 'output全球/'+csdata+'_grid_预测全球有效值索引.csv', valid_grid_loc, delimiter=',')

