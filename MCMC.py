import numpy as np
import os
import time

def GenerateParamValues2(c_op):
    flag = True
    while (flag):
        D = 5
        randVector = np.random.uniform(-0.5, 0.5, paramNum)  # 均匀分布伪随机数
        cNew = c_op + randVector * (cmax - cmin) / D
        if (isQualified(cNew)):
            flag = False
    return cNew


def GenerateParamValues(c_op):
    flag = True
    while (flag):
        randVector = np.random.randn(paramNum)  # 正态分布伪随机数
        cT = randVector * np.sqrt(eigD)
        cNew = np.dot(eigV, (np.dot(eigV.T, c_op) + cT))
        if (isQualified(cNew)):
            flag = False
    return cNew


def isQualified(cNew):
    flag = True
    for i in range(paramNum):
        if (cNew[i] > cmax[i] or cNew[i] < cmin[i]):
            flag = False
            break
    return flag

def ku7(y, cNew):
    litter, mbc = y
    vmax = cNew[0]
    km = cNew[1]
    cue = cNew[2]
    kb = cNew[3]
    kl = cNew[4]
    out = [-vmax*mbc*litter/(km+litter)-kl*litter,
           cue*(vmax*mbc*litter/(km+litter))-kb*mbc]
    return out


def run_model(cNew,Litterfall):
    x = np.zeros([2, Nt + 1], dtype=float)

    x[:, 0] = x0
    ## simulate for 10 years
    for i in range(1, Nt + 1):
        outku7 = ku7(x[:, i-1], cNew)
        x[:, i] = outku7 + x[:, i-1]

    ## 模拟碳库，2行，Nt列
    xsimu = x[:, range(1, Nt + 1)]
    ## 获取对应时间点的值
    tt = timepoint.astype(int)-1 #index从0开始
    litter_simu = xsimu[:, tt]

    vmax = cNew[0]
    km = cNew[1]
    cue = cNew[2]
    kb = cNew[3]
    kl = cNew[4]

    L_ku = kb*km/(cue*vmax-kb)
    mbc_ku = (Litterfall-kl*L_ku)*(km + L_ku)/(vmax*L_ku)
    deerta_L= Litterfall - vmax*mbc_ku*L_ku/(km + L_ku) - kl * L_ku
    deerta_MBC = cue * vmax*mbc_ku*L_ku/(km + L_ku) - kb * mbc_ku
    SOCin = kl * L_ku + kb *mbc_ku
    Out2 = (deerta_L,deerta_MBC,SOCin, L_ku)

    return litter_simu,Out2


def getBest():
    bestId = np.where(J_record == np.nanmin(J_record[0:record]))
    bestC = c_record[:, bestId[0][0]]
    bestSimu,temp = run_model(bestC,0.53)
    return [bestC, bestSimu]

def write_io_file2(outDir,namee):
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_mismatch_accepted_text.txt', J_record[1:record])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_param_accepted_text.txt', c_record[:, 1:record])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_accepted_num_text.txt', [record])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_bestParam_text.txt', bestC)
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_litter_bestSimu_text.txt', bestSimu[0])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_MBC_bestSimu_text.txt', bestSimu[1])

def write_io_file(outDir,namee):
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_mismatch_accepted.txt', J_record[1:record])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_param_accepted.txt', c_record[:, 1:record])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_accepted_num.txt', [record])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_bestParam.txt', bestC)
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_litter_bestSimu.txt', bestSimu[0])
    np.savetxt(outDir + '/' + namee[0] + '_' + namee[1] + '_' + namee[2] +'_MBC_bestSimu.txt', bestSimu[1])


if __name__ == '__main__':
    start = time.perf_counter()

    outDir = r'input data dir XXX'
    Nt = 3900  # 10.5年

    ## 初始化
    cmin = np.array([1,10,0.02,0,0])
    cmax = np.array([100,4000,0.3,0.01,0.01])
    c = np.array([5.77,2500,0.15,0.01,0.001])
    paramNum = 5  # number of parameters

    ##数据集及其方差
    datafile = open("input data", "r")  #  txt file
    data = datafile.readlines()
    datafile.close()

    for  dd in data:
        temp_data = dd.split(' ', 3)
        temp_data2 = temp_data[3].split()
        temp_data2 = list(map(float, temp_data2))
        
        aa = temp_data2[2]
        temp_data3 = temp_data2[int(aa + 3):int(aa * 2 + 3)]
        timepoint = temp_data2[3:int(aa + 3)]
        timepoint = np.array(timepoint)
        litterfall = temp_data2[int(aa * 3 + 3)]
        SOCinput = temp_data2[int(aa * 3 + 4)]
        CMSlitter = temp_data2[int(aa * 3 + 5)] / 2 
        
        obsList = [x * temp_data2[0]*0.45*25 / 100 for x in temp_data3]
        ## 方差
        varList = np.var(obsList, ddof=1)

        # 状态初始值
        x0 = [temp_data2[0]*0.45*25, 0.0023 * temp_data2[0]*0.45*25]

        # 控制条件
        obsList.extend((0, 0, SOCinput, CMSlitter))
        print(dd)

        # 检查是否存在文件
        if (
        os.path.isfile(outDir + '/' + temp_data[0] + '_' + temp_data[1] + '_' + temp_data[2] + '_param_accepted_text.txt')):
            print('已经存在测试运行参数文件！')

        else:
            #
            nsimu = 10000  # 迭代次数
            record = 0  # 接受参数
            c_record = np.zeros((paramNum, nsimu + 1), dtype=float)  # 保存参数
            c_record[:, 0] = c
            J_record = np.zeros(nsimu + 1, dtype=float)  # 保存误差
            J_record[0] = 300000
            for simu in range(nsimu):
                # 根据当前接受的参数值生成一组新的参数值
                c_new = GenerateParamValues2(c_record[:, record])

                # running model
                simuList,out2 = run_model(c_new,litterfall)
                simuList2 = simuList[0]
                simuList3 = np.append(simuList2, out2)

                # 成本函数
                J_new = sum([sum((simuList3 - obsList) ** 2) / (2 * varList)])
                delta_J = J_aarecord[record] - J_new
                ## Moving step: to decide whether the new set of parameter values will be accepted or not
                # 是否接受参数
                randNum = np.random.uniform(0, 1, 1) 
                if (min(1.0, np.exp(delta_J)) > randNum):
                    record += 1
                    c_record[:, record] = c_new  # 接受参数
                    J_record[record] = J_new  # 保存误差

            [bestC, bestSimu] = getBest()
            # 写入文件
            write_io_file2(outDir, temp_data[0:3])



        #正式模拟
        if (
        os.path.isfile(outDir + '/' + temp_data[0] + '_' + temp_data[1] + '_' + temp_data[2] + '_param_accepted.txt')):
            print('已经存在正式运行参数文件！')

        else:
            param_test = np.loadtxt(
                outDir + '/' + temp_data[0] + '_' + temp_data[1] + '_' + temp_data[2] + '_param_accepted_text.txt')
            cov_test = np.cov(param_test)  
            eig = np.linalg.eig(cov_test)
            eigD = eig[0]
            eigV = eig[1]

            nsimu = 12000  
            record = 0
            c_record = np.zeros((paramNum, nsimu + 1), dtype=float)  
            c_record[:, 0] = c
            J_record = np.zeros(nsimu + 1, dtype=float) 
            J_record[0] = 300000
            for simu in range(nsimu):
                c_new = GenerateParamValues(c_record[:, record])

                # running model
                simuList, out2 = run_model(c_new, litterfall)
                simuList2 = simuList[0]
                simuList3 = np.append(simuList2, out2)

                #
                J_new = sum([sum((simuList3 - obsList) ** 2) / (2 * varList)])
                delta_J = J_record[record] - J_new

                randNum = np.random.uniform(0, 1, 1)  
                if (min(1.0, np.exp(delta_J)) > randNum):
                    record += 1
                    c_record[:, record] = c_new 
                    J_record[record] = J_new  
            [bestC, bestSimu] = getBest()
            write_io_file(outDir, temp_data[0:3])

    end = time.perf_counter()
    print("final is in ", end - start)
