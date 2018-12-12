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
"""
def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)  # normalize


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


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