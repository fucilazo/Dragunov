"""
MCF(Monte-Carlo Particle Filter)
Copyright (c) 2018 Takuma Sakaki
This module is released under the MIT License.
http://opensource.org/licenses/mit-license.php
"""

####################################
###モンテカルロフィルタ(MCF)クラス
####################################

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.style.use('fivethirtyeight')
from numpy.random import rand, normal


class MCF():
    system_equation = None
    obs_equation = None
    system_equation_gen = None
    obs_L = None
    total_step = None
    x_0 = None

    x = None
    y = None
    sim_x = np.array(None)

    def __init__(self, system_equation, obs_equation, system_equation_gen, obs_L, x_0, total_step):
        self.system_equation = system_equation
        self.obs_equation = obs_equation
        self.system_equation_gen = system_equation_gen
        self.obs_L = obs_L
        self.x_0 = x_0
        self.total_step = total_step

    def Set_totalstep(self, total_step):
        self.total_step = total_step

    def Set_obs(self, y):
        if len(y) != self.total_step + 1:
            print("the expected length of y is", (self.total_step + 1), "but the length of inputed y is", len(y))
        else:
            self.y = y

    def Set_truestates(self, x):
        self.x = x

    def Get_results(self):
        return self.sim_x

    # データ生成メソッド
    def DataGenerate(self, true_x_0=None, plot=False):

        if (true_x_0 == None):
            true_x_0 = input("Input the value of x_0: ")

        T = self.total_step
        x = np.empty(T + 1)
        y = np.empty(T + 1)
        x[0] = true_x_0
        y[0] = None
        for i in range(1, T + 1):
            x[i] = self.system_equation(x[i - 1], i)
            y[i] = self.obs_equation(x[i])

        self.x = x
        self.y = y

        if plot == True:
            plt.plot(x, color='green', label='true-state')
            plt.plot(y, color='blue', label='observed-value')
            plt.xlabel("Time")
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5, 0.5, .100), borderaxespad=0.)

    # Calculation Aggregated Values
    def Get_summary(self, Per_CI=0.95, plot=False):

        if Per_CI <= 0 or Per_CI > 1:
            print("Per_CI should be [0,1].")

        if self.sim_x is None:
            print("This Method can be used only after calculation")
            sys.exit()

        T = self.total_step
        m = len(self.x_0)  # mは粒子の個数
        ave_x = np.full(T + 1, 100, dtype=np.float)
        mid_x = np.full(T + 1, 100, dtype=np.float)
        lower_CI = np.full(T + 1, 100, dtype=np.float)
        upper_CI = np.full(T + 1, 100, dtype=np.float)
        low = int(m * (1 - Per_CI) / 2) - 1
        high = int(m - low) - 1

        for i in range(T + 1):
            tmp = self.sim_x[i]
            x_i = np.sort(tmp)
            ave_x[i] = np.average(x_i)
            mid_x[i] = np.median(x_i)
            lower_CI[i] = x_i[low]
            upper_CI[i] = x_i[high]

        if (plot == True):
            # 結果のプロット
            plt.plot(self.x, color='green', linewidth=2, label="true-state")
            plt.plot(ave_x, color='red', linewidth=2, label="estimated-state")
            plt.plot(upper_CI, '--', color='red', linewidth=2, label="upper-CI")
            plt.plot(lower_CI, '--', color='red', linewidth=2, label="lower-CI")
            plt.xlabel("Time")
            plt.legend(loc='upper right', bbox_to_anchor=(1.05, 0.5, 0.5, .100), borderaxespad=0.)

        return {'ave_x': ave_x, 'mid_x': mid_x, 'lower_CI': lower_CI, 'upper_CI': upper_CI}

    # 平均二乗誤差
    def MSE(self, estimation_method="average"):
        if self.sim_x is None:
            print("This Method can be used only after calculation")
            sys.exit()
        if self.x is None:
            print("This Method can be used only after defining true-state")
            sys.exit()

        MSE = 0
        T = self.total_step

        for t in range(1, T + 1):
            if estimation_method == "average":
                est_x = np.average(self.sim_x[t])
            elif estimation_method == "median":
                est_x = np.median(self.sim_x[t])
            else:
                print("error: estimation_method:", estimation_method)
                sys.exit()

            MSE = MSE + ((self.x[t] - est_x) ** 2) / T

        return MSE

    # Total Calculation
    def Filtering(self, resampling_method="original", smoothing_lag=0):

        # 変数チェックとセッティング
        if self.y is None:
            print("observed value (y) is not defined. Define it with the methods 'set_obs' or 'DataGenerate'")
        if resampling_method != "original" and resampling_method != "stratified":
            print("original or stratified is only usable as resampling_method")
        T = self.total_step
        self.sim_x = np.empty((T + 1, len(self.x_0)))

        # 初期値
        x_samples = self.x_0
        self.sim_x[0] = x_samples

        # Main calc
        for i in tqdm(range(1, T + 1)):
            x_samples = self.__Step(x_samples, i, resampling_method, smoothing_lag)

        print("Simulation finished successfully.")

    # 1step
    def __Step(self, x_samples, t, resampling_method, smoothing_lag):

        m = len(x_samples)

        # RandomGenerate
        x_samples = self.system_equation_gen(x_samples, t)

        # likelihood
        y_t_array = np.full(m, self.y[t])
        w_t = self.obs_L(x_samples, y_t_array)

        # Resampling -- with high speed
        new_samples = np.empty(m)
        if resampling_method == "original":
            u_t = rand(m)
        elif resampling_method == "stratified":
            u_t = rand(m) / m + np.linspace(0, 1, m + 1)[:-1]

        w_t = w_t / np.sum(w_t)
        w_t = w_t.cumsum()

        for i in range(m):
            u = u_t[i]
            try:
                j = np.where(w_t > u)[0][0]
            except:
                print(u, "is a unexpected number. plz debug accessing self.w")
                self.w = w_t
                sys.exit()

            new_samples[i] = x_samples[j]

            # Smoothing
            if smoothing_lag > 0:
                for past_time in range(max(0, t - smoothing_lag), t):
                    self.sim_x[past_time][i] = self.sim_x[past_time][j]

        self.sim_x[t] = new_samples

        return new_samples


# テスト,線形ガウス型状態空間モデル
if __name__ == "__main__":
    print("Conducting test program...")


    # システム方程式、観測方程式は以下。
    def system_equation_1(x, t):
        v = normal(0, 1, 1)
        return x + v


    def obs_equation_1(x):
        e = normal(0, 0.5)
        return x + e


    # 予測分布の発生式
    def system_equation_gen_1(x_samples, t):
        v = normal(0, 1, len(x_samples))
        return x_samples + v


    # 尤度の計算式
    def obs_L_1(x, y):
        t = y - x
        return (1 / np.sqrt(2 * np.pi * 0.25)) * np.exp(-1 * (t ** 2) / 0.5)


    # 10000個の0の値の粒子から、50ステップのシミュレーションを行う。
    MCF = MCF(system_equation_1, obs_equation_1, system_equation_gen_1, obs_L_1, np.zeros(10000), 50)

    # 初期値x_0を0として、T=50までの真の状態x_tと観測値y_tを発生させる
    MCF.DataGenerate(0)
    # 計算を行う
    MCF.Filtering()
    summary = MCF.Get_summary(plot=True)

"""
Reference:
Kitagawa G (1996) Monte Carlo filter and smoother for non-Gaussian nonlinear state space models. J Comput Graph Stat 5:1–25
矢野浩一（2014）粒子フィルタの基礎と応用：フィルタ・平滑化・パラメータ推定
"""