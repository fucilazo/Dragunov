import numpy as np
import scipy.stats
from numpy.random import uniform, randn, random
import matplotlib.pyplot as plt

"""
步骤1：生成初始粒子

为了估计机器人真实位置，我们选取机器人位置x,y和其朝向θ作为状态变量，因此每个粒子需要有三个特征。用矩阵particles存储N个粒子信息，其为N行3列的矩阵。
下面的代码可以在一个区域上生成 服从均匀分布的粒子 or 服从高斯分布的粒子：
"""
def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


"""
步骤2：利用系统模型预测状态

每一个粒子都代表了一个机器人的可能位置，假设我们发送控制指令让机器人移动0.1米,转动0.7弧度，我么可以让每个粒子移动相同的量。
由于控制系统存在噪声，因此需要添加合理的噪声。
"""
def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)"""

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist


"""
步骤3：更新粒子权值

使用测量数据(机器人距每个路标的距离)来修正每个粒子的权值，保证与真实位置越接近的粒子获得的权值越大。
由于机器人真实位置是不可测的，可以看作一个随机变量。
根据贝叶斯公式可以称机器人在位置x处的概率P(x)为先验分布密度(prior)，或预测值，预测过程是利用系统模型(状态方程)预测状态的先验概率密度，也就是通过已有的先验知识对未来的状态进行猜测。
在位置x处获得观测量z的概率P(z|x)为似然函数(likelihood)。后验概率为P(x|z)， 即根据观测值z来推断机器人的状态x。
更新过程利用最新的测量值对先验概率密度进行修正，得到后验概率密度，也就是对之前的猜测进行修正。

p(x|z) = p(z|x)p(x) / p(z) = likelihoodxprior / normalization
贝叶斯定理中的分母P(z)不依赖于x，因此，在贝叶斯定理中因子P(z)-1常写成归一化因子η。

可以看出，在没有测量数据可利用的情况下，只能根据以前的经验对x做出判断，即只是用先验分布P(x)；
但如果获得了测量数据，则可以根据贝叶斯定理对P(x)进行修正，即将先验分布与实际测量数据相结合得到后验分布。
后验分布是测量到与未知参数有关的实验数据之后所确定的分布，它综合了先验知识和测量到的样本知识。
因此基于后验分布对未知参数做出的估计较先验分布而言更有合理性，其估计的误差也较小。
这是一个将先验知识与测量数据加以综合的过程，也是贝叶斯理论具有优越性的原因所在。

------------------------------------------------------------------------------------------------------------------------
贝叶斯滤波递推的基本步骤如下：

Algorithm Bayes_filter(bel(x[t-1], u[t], z[t])):
    for all x[t] do
        mean(bel(x[t])) = ∫{ p(x[t]|u[t], x[t-1])bel(x[t-1]) }dx  # prediction
        bel(x[t]) = η{ p(z[t]|x[t])bel(x[t]) }  # correction
    endfor
    return bel(x[t])

式子中概率p(xt|ut,xt-1)是机器人的状态转移概率，它描述了机器人在控制量ut的作用下从前一个状态xt-1转移到下一状态xt的概率( It describes how much the x changes over one time step)。
由于真实环境的复杂性（比如控制存在误差、噪声或模型不够准确），机器人根据理论模型推导下一时刻的状态时，只能由概率分布p(xt|ut,xt-1)来描述，而非准确无误的量。
参考Probabilistic Robotics中的一个例子：t-1时刻门处于关闭状态，然后在控制量Ut的作用下机器人去推门，则t时刻门被推开的概率为0.8，没有被推开的概率为0.2

然而一般情况下，概率p(xt|ut,xt-1)及p(zt|xt)的分布是未知的，仅对某些特定的动态系统可求得后验概率密度的解析解。
比如当p(x0)，p(xt|ut,xt-1)及p(zt|xt)都为高斯分布时，后验概率密度仍为高斯分布，并可由此推导出标准卡尔曼滤波。
------------------------------------------------------------------------------------------------------------------------
"""
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)  # normalize


"""
步骤4：重采样

在计算过程中，经过数次迭代，只有少数粒子的权值较大，其余粒子的权值可以忽略不计，粒子权值的方差随着时间增大，状态空间中的有效粒子数目减少，这一问题称为权值退化问题。
随着无效粒子数目的增加，大量计算浪费在几乎不起作用的粒子上，使得估计性能下降。通常采用有效粒子数Neff衡量粒子权值的退化程度。Neff的近似计算公式为：
                                                        Neff = 1/∑(w^2)
有效粒子数越小，表明权值退化越严重。当Neff的值小于某一阈值时，应当采取重采样措施，根据粒子权值对离散粒子进行重采样。
重采样方法舍弃权值较小的粒子，代之以权值较大的粒子，有点类似于遗传算法中的“适者生存”原理。
重采样的方法包括多项式重采样(Multinomial resampling)、残差重采样(Residual resampling)、分层重采样(Stratified resampling)和系统重采样(Systematic resampling)等。
重采样带来的新问题是，权值越大的粒子子代越多，相反则子代越少甚至无子代。这样重采样后的粒子群多样性减弱，从而不足以用来近似表征后验密度。
克服这一问题的方法有多种，最简单的就是直接增加足够多的粒子，但这常会导致运算量的急剧膨胀。其它方法可以去查看有关文献，这里暂不做介绍。

简单的多项式重采样(Multinomial resampling)代码如下：
这一步是非常关键的一步，即如何根据预测值和测量值修正粒子的权值。函数update的参数particles可以看做先验概率分布p(xt|ut,xt-1)的取样值。
假设传感器噪声符合高斯分布，那么在位置xt处获得观测量zt的概率p(zt|xt)可以由scipy.stats.norm(dist, R).pdf(z[i])来描述。
由于多个路标特征之间的观测相互独立，因此符合概率乘法公式的条件，即下面代码第5行中的累乘。
"""
def neff(weights):
    return 1. / np.sum(np.square(weights))


def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)  # normalize


"""
步骤5：计算状态变量估计值

系统状态变量估计值可以通过粒子群的加权平均值计算出来。
"""
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def run_pf(N, iters=18, sensor_std_err=0.1, xlim=(0, 20), ylim=(0, 20)):
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    NL = len(landmarks)

    # create particles and weights
    particles = create_uniform_particles((0, 20), (0, 20), (0, 2 * np.pi), N)
    weights = np.zeros(N)

    xs = []  # estimated values
    robot_pos = np.array([0., 0.])

    for x in range(iters):
        robot_pos += (1, 1)

        # distance from robot to each landmark
        zs = np.linalg.norm(landmarks - robot_pos, axis=1) + (randn(NL) * sensor_std_err)

        # move particles forward to (x+1, x+1)
        predict(particles, u=(0.00, 1.414), std=(.2, .05))

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)

        # resample if too few effective particles
        if neff(weights) < N / 2:
            simple_resample(particles, weights)

        # Computing the State Estimate
        mu, var = estimate(particles, weights)
        xs.append(mu)

    xs = np.array(xs)
    plt.plot(np.arange(iters + 1), 'k+')
    plt.plot(xs[:, 0], xs[:, 1], 'r.')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], alpha=0.4, marker='o', c=randn(4), s=100)  # plot landmarks
    plt.legend(['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim([-2, 20])
    plt.ylim([0, 22])
    print('estimated position and variance:\n\t', mu, var)
    plt.show()


if __name__ == '__main__':
    run_pf(N=5000)