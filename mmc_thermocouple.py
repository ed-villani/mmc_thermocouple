import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

constants = np.array(
    [
        0.0,
        3.8783277 * 10 ** (-2),
        -1.1612344 * 10 ** (-6),
        6.9525655 * 10 ** (-11),
        -3.0090077 * 10 ** (-15),
        8.8311584 * 10 ** (-20),
        -1.6213839 * 10 ** (-24),
        1.6693362 * 10 ** (-29),
        -7.3117540 * 10 ** (-35)
    ]
)

measurements = [
    np.array([2412, 2408, 2412, 2412, 2414]),
    np.array([25382, 25377, 25350, 25539, 25340]),
    np.array([36245, 36248, 36248, 36251, 36253]),
]


def m_factor(p):
    return round((10 ** 4) * (1 / (1 - p)))


def thermocouple_reference_function(voltage):
    return sum([c * voltage ** i for i, c in enumerate(constants)])


def sensitivity_coefficient(voltage):
    return sum([i * c * voltage ** (i - 1) for i, c in enumerate(constants)])


def thermocouple(temperature, cs, cs0, calibration, resolution, parasite_voltage, ref_temperature, tol_temperature,
                 grd_temperature):
    return temperature + cs * (calibration + resolution + parasite_voltage) - (cs / cs0) * ref_temperature + \
           tol_temperature + grd_temperature


def gen_calibration(M):
    return np.random.normal(0, 1, M)


def gen_resolution(M):
    return np.random.uniform(-0.5, 0.5, M)


def gen_parasite_voltage(M):
    return np.random.normal(0, 2, M)


def gen_ref_temperature(M):
    return np.random.normal(0, 0.1, M)


def gen_grd_temperature(M):
    return np.random.uniform(-11, +11, M)


def random_samples(data, M):
    mean = np.mean(data)
    std = np.std(data)
    return np.random.normal(mean, std, M)


def gen_tol_temperature(data, M):
    import copy
    aux_data = copy.deepcopy(data)
    N = len(np.where(aux_data > 375)[0])
    del aux_data
    return np.concatenate([np.random.normal(0, 1.5, N), np.random.normal(0, 0.4, M - N)])


def limits(data, p, M):
    import copy

    q = int(M * p)
    r = (M - q) / 2

    aux_data = copy.deepcopy(data)
    aux_data.sort()

    r_inf = 1
    for w in range(M - q):
        if (aux_data[w + q] - aux_data[w] <= aux_data[r_inf + q] - aux_data[r_inf]):
            r_inf = w

    lim_sup = aux_data[r_inf]
    lim_inf = aux_data[r_inf + q]

    return lim_sup, lim_inf


def plot_pdf(mmcs, name):
    fig, axes = plt.subplots(1, len(mmcs), figsize=(plt.figaspect(0.3)))
    for i, data in enumerate(mmcs):
        ax = axes[i]
        ax.set_title(f"PDF {name[i]}")
        inf, sup = limits(data, 0.95, len(data))
        ax.axvline(inf, 0, 1, linestyle='--', c='red')
        ax.axvline(sup, 0, 1, linestyle='--', c='red')
        ax.axvline(np.mean(data), 0, 1, linestyle='--', c='black')
        # ax.axvline(np.mean(data) + 2*np.std(data), 0, 1, linestyle='--', c='black')
        # ax.axvline(np.mean(data) - 2*np.std(data), 0, 1, linestyle='--', c='black')
        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel("Densidade")
        sb.histplot(data, kde=True, element="step", stat="density", ax=ax)

    plt.show()


def plot_cdf(mmcs, name):
    fig, axes = plt.subplots(1, 3, figsize=(plt.figaspect(0.3)))
    for i, data in enumerate(mmcs):
        ax = axes[i]
        ax.set_title(f"CDF {name[i]}")
        ax.set_xlabel("Temperatura (°C)")
        ax.set_ylabel("Densidade Acumulada")
        sb.histplot(data, kde=True, cumulative=True, element="step", stat="density", ax=ax)

    plt.show()


def info(data, name, p):
    inf, sup = limits(data, p, len(data))
    print(f'''
        For MMC {name}:
        M = {m_factor(p)}
        Result: {np.mean(data)} +\- {np.std(data)} °C
        Inferior Limit: {inf} °C
        Superior Limit: {sup} °C
        Meadian: {np.median(data)} °C
    ''')


def mmc_thermocouple(data, M):
    voltages = random_samples(data, M)
    temperature = thermocouple_reference_function(voltages)
    cs = sensitivity_coefficient(voltages)
    return thermocouple(
        temperature=temperature,
        cs=cs,
        cs0=constants[1],
        calibration=gen_calibration(M),
        resolution=gen_resolution(M),
        parasite_voltage=gen_parasite_voltage(M),
        ref_temperature=gen_ref_temperature(M),
        tol_temperature=gen_tol_temperature(temperature, M),
        grd_temperature=gen_grd_temperature(M)
    )


def main():
    M = m_factor(0.95)
    mmcs = [mmc_thermocouple(m, M) for m in measurements]

    titles = ['5 min', '30 min', '60 min']
    plot_pdf(mmcs, titles)
    plot_cdf(mmcs, titles)
    [info(mmc, title, 0.95) for mmc, title in zip(mmcs, titles)]


if __name__ == '__main__':
    main()
