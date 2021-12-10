import simpy
import numpy as np
from math import factorial
from matplotlib import pyplot as plt


class SMO(object):
    def __init__(self, env, number_channels):
        self.env = env
        self.wait_times = []
        self.queue_list = []
        self.smo_list = []
        self.queue_time = []
        self.cancel_list = []
        self.loader = simpy.Resource(env, number_channels)

    def cancel(self, visitors, serv_flow):
        yield self.env.timeout(np.random.exponential(1 / serv_flow))

    def wait(self, visitors, queue_flow):
        yield self.env.timeout(np.random.exponential(1 / queue_flow))


def create_SMO(number_channels, app_flow, serv_flow, queue_flow, queue_length, test_time):
    env = simpy.Environment()
    smo = SMO(env, number_channels)
    env.process(start_SMO(env, smo, number_channels, app_flow, serv_flow, queue_flow, queue_length))
    env.run(until=test_time)
    return smo.wait_times, smo.queue_list, smo.queue_time, smo.cancel_list, smo.smo_list


def start_SMO(env, smo, number_channels, app_flow, serv_flow, queue_flow, queue_length):
    visitors = 0
    while True:
        yield env.timeout(np.random.exponential(1 / app_flow))
        env.process(operation(env, visitors, smo, serv_flow, queue_flow, queue_length, number_channels))
        visitors = visitors + 1


def operation(env, visitors, smo, serv_flow, queue_flow, queue_length, num_channel):
    len_queque_global = len(smo.loader.queue)
    count_active_channel_global = smo.loader.count
    with smo.loader.request() as request:
        len_queque = len(smo.loader.queue)
        count_active_channel = smo.loader.count
        smo.queue_list.append(len_queque_global)
        smo.smo_list.append(len_queque_global + count_active_channel_global)
        if len_queque <= queue_length:
            smo.cancel_list.append(count_active_channel + len_queque)
            arrival_time = env.now
            result = yield request | env.process(smo.wait(visitors, queue_flow))
            smo.queue_time.append(env.now - arrival_time)
            if request in result:
                yield env.process(smo.cancel(visitors, serv_flow))
                smo.wait_times.append(env.now - arrival_time)
            else:
                smo.wait_times.append(env.now - arrival_time)
        else:
            smo.cancel_list.append(queue_length + num_channel + 1)
            smo.queue_time.append(0)
            smo.wait_times.append(0)


def average_number_applications(queue_list):
    average_queue_list = np.array(queue_list).mean()
    print("Average number of applications in the queue: ", average_queue_list)
    return average_queue_list


def average_smo_number_applications(smo_list):
    average_smo_list = np.array(smo_list).mean()
    print("Average number of applications served in the SMO: ", average_smo_list)
    return average_smo_list


def average_queue_application_time(queue_time):
    average_queue_time = np.array(queue_time).mean()
    print("Average time spent by an application in the queue: %s" % (average_queue_time))
    return average_queue_time


def average_smo_application_time(wait_times):
    average_smo_time = np.array(wait_times).mean()
    print("Average time spent by an application in the CMO: %s " % (average_smo_time))
    return average_smo_time


def theoretic_probabilities(number_channels, queue_length, app_flow, serv_flow, queue_flow):
    print("_______________________Theoretic_______________________")
    ro = app_flow / serv_flow
    betta = queue_flow / serv_flow
    final_propability = []
    sum_probabilities = 0
    p0 = (sum([ro ** i / factorial(i) for i in range(number_channels + 1)]) +
          (ro ** number_channels / factorial(number_channels)) *
          sum([ro ** i / (np.prod([number_channels + t * betta for t in range(1, i + 1)])) for i in range(1, queue_length + 1)])) ** -1
    print('Theoretical P0:', p0)
    final_propability.append(p0)
    sum_probabilities += p0
    for i in range(1, number_channels + 1):
        px = (ro ** i / factorial(i)) * p0
        sum_probabilities += px
        final_propability.append(px)
        print(f'Theoretical p{i}: {px}')
    pn = px
    p_queque = px
    for i in range(1, queue_length + 1):
        px = (ro ** (i) / np.prod([number_channels + t * betta for t in range(1, i + 1)])) * pn
        sum_probabilities += px
        if i < queue_length:
            p_queque += px
        print(f'Theoretical p{number_channels + i}: {px}')
        final_propability.append(px)
    p = px
    print(f'Theoretical probability of reject: {p}')
    final_propability.append(p)
    print("Theoretical probability of queuing: ", p_queque)
    final_propability.append(p_queque)
    relative_t = 1 - p
    print("Theoretical relative throughput: ", relative_t)
    absolute_t = relative_t * app_flow
    final_propability.append(absolute_t)
    print("Theoretical absolute throughput: ", absolute_t)
    n_people_queque = sum([i * pn * (ro ** i) / np.prod([number_channels + l * betta for l in range(1, i + 1)]) for
                           i in range(1, queue_length + 1)])
    print("Average applications in queue: ", n_people_queque)
    K_av = sum([index * p0 * (ro ** index) / factorial(index) for index in range(1, number_channels + 1)]) + sum(
        [(number_channels + index) * pn * ro ** index / np.prod(
            np.array([number_channels + l * betta for l in range(1, index + 1)])) for
         index in range(1, queue_length + 1)])
    print("Average applications in SMO: ", K_av)
    n_average = relative_t * ro
    print("Average number of full channels: ", n_average)
    T_queque = n_people_queque / app_flow
    final_propability.append(T_queque)
    print("Average application time in queue: ", T_queque)
    T_smo = K_av / app_flow
    final_propability.append(T_smo)
    print("Average application time in SMO: ", T_smo)
    return final_propability


def empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, number_channels, queue_length, app_flow,
                          serv_flow):
    print("_______________________Empiric_______________________")
    cancel_array = np.array(cancel_list)
    p_x = []
    p_queue = []
    for i in range(1, number_channels + queue_length + 2):
        p_x.append(len(cancel_array[cancel_array == i]) / len(cancel_array))
        if i > number_channels and i <= number_channels + queue_length:
            p_queue.append(len(cancel_array[cancel_array == i]) / len(cancel_array))
    p_fail = len(cancel_array[cancel_array == (number_channels + queue_length + 1)]) / len(cancel_array)
    final_propability = []
    p_queued = sum(p_queue)
    for i, item in enumerate(p_x):
        print(f'Empiric P{i}: {item}')
        final_propability.append(item)
    print("Empiric probability of reject ", p_fail)
    final_propability.append(p_fail)
    print("Empiric probability of queuing: ", p_queued)
    final_propability.append(p_queued)
    relative_t = 1 - p_fail
    print("Empiric relative throughput: ", relative_t)
    absolute_t = relative_t * app_flow
    final_propability.append(absolute_t)
    print("Empiric absolute throughput: ", absolute_t)
    n_people_queque = average_number_applications(queue_list)
    number_av = average_smo_number_applications(smo_list)
    n_average = relative_t * app_flow / serv_flow
    print("Average number of full channels: ", n_average)
    time_queue = average_queue_application_time(queue_time)
    final_propability.append(time_queue)
    time_smo = average_smo_application_time(wait_times)
    final_propability.append(time_smo)
    return final_propability


def show_difference(empirics, theoretics):
    fig, ax = plt.subplots()
    ax.bar([i - 0.2 for i in range(len(empirics))], empirics, tick_label=["p0", "p1", "p2", "p3", "p4", "p_reject", "p_queue", "absolute_throughput", "time_queue", "time_smo"], width=0.4, color='blue',
           label='Empiric')
    ax.bar([i + 0.2 for i in range(len(empirics))], theoretics, tick_label=["p0", "p1", "p2", "p3", "p4", "p_reject", "p_queue", "absolute_throughput", "time_queue", "time_smo"], width=0.4, color='red', label='Theoretical')
    ax.set_title('Empiric and theoretic difference')
    fig.set_figwidth(15)
    fig.set_figheight(6)
    plt.legend()
    plt.show()


def plot_queue_system_probabilities(counts, theoretical_probabilities, interval_count):
  intervals = np.array_split(counts, interval_count)
  for i in range(1, len(intervals)):
    intervals[i] = np.append(intervals[i], intervals[i - 1])
  for i in range(len(theoretical_probabilities)):
    if i < n+m+1:
        interval_probabilities = []
        for interval in intervals:
          interval_probabilities.append(len(interval[interval == i]) / len(interval))
        plt.figure(figsize=(5, 5))
        plt.bar(range(len(interval_probabilities)), interval_probabilities)
        plt.title(f"Probabilitiy {i}")
        plt.axhline(y = theoretical_probabilities[i], xmin=0, xmax=len(interval_probabilities), color='purple')
        plt.show()


def show_changes(empirics1, empirics2, empirics3, value_name, values):
    fig, ax = plt.subplots()
    labels = ["P[0]", 'P[1]', 'P[2]', 'P[3]', 'P[4]']
    label1 = value_name + " = " + str(values[0])
    ax.bar([i for i in range(len(empirics1))], empirics1, tick_label=labels, width=0.3, color='blue',
           label=label1)
    label2 = value_name + " = " + str(values[1])
    ax.bar([i + 0.3 for i in range(len(empirics1))], empirics2, tick_label=labels, width=0.3, color='yellow',
           label=label2)
    label3 = value_name + " = " + str(values[2])
    ax.bar([i + 0.6 for i in range(len(empirics1))], empirics3, tick_label=labels, width=0.3, color='red',
           label=label3)
    ax.set_title('Empiric values changes')
    fig.set_figwidth(19)
    fig.set_figheight(6)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    n = 2
    u = 1
    ro = 1
    m = 2

values = [1, 4, 7]
lambd = values[0]
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)

empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
show_difference(empiric_probability, theoretic_probability)
plot_queue_system_probabilities(smo_list, theoretic_probability, 1000)
empirics1 = []
empirics1.append(empiric_probability[0])
empirics1.append(empiric_probability[1])
empirics1.append(empiric_probability[2])
empirics1.append(empiric_probability[3])
empirics1.append(empiric_probability[4])

lambd = values[1]
empiric_probability.clear()
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics2 = []
empirics2.append(empiric_probability[0])
empirics2.append(empiric_probability[1])
empirics2.append(empiric_probability[2])
empirics2.append(empiric_probability[3])
empirics2.append(empiric_probability[4])

lambd = values[2]
empiric_probability.clear()
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics3 = []
empirics3.append(empiric_probability[0])
empirics3.append(empiric_probability[1])
empirics3.append(empiric_probability[2])
empirics3.append(empiric_probability[3])
empirics3.append(empiric_probability[4])


show_changes(empirics1, empirics2, empirics3, "Lambda", values)

#---------------------------------------------------------------------------------------------------------
lambd = 2
empirics1.clear()
empirics3.clear()
empirics2.clear()
empiric_probability.clear()
values.clear()
values = [1, 4, 7]

u = values[0]
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics1 = []
empirics1.append(empiric_probability[0])
empirics1.append(empiric_probability[1])
empirics1.append(empiric_probability[2])
empirics1.append(empiric_probability[3])
empirics1.append(empiric_probability[4])


u = values[1]
empiric_probability.clear()
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics2 = []
empirics2.append(empiric_probability[0])
empirics2.append(empiric_probability[1])
empirics2.append(empiric_probability[2])
empirics2.append(empiric_probability[3])
empirics2.append(empiric_probability[4])


u = values[2]
empiric_probability.clear()
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics3 = []
empirics3.append(empiric_probability[0])
empirics3.append(empiric_probability[1])
empirics3.append(empiric_probability[2])
empirics3.append(empiric_probability[3])
empirics3.append(empiric_probability[4])


show_changes(empirics1, empirics2, empirics3, "u", values)


#---------------------------------------------------------------------------------------------------------
u = 2
empirics1.clear()
empirics3.clear()
empirics2.clear()
empiric_probability.clear()
values.clear()
values = [2, 3, 4]

ro = values[0]
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics1 = []
empirics1.append(empiric_probability[0])
empirics1.append(empiric_probability[1])
empirics1.append(empiric_probability[2])
empirics1.append(empiric_probability[3])
empirics1.append(empiric_probability[4])


ro = values[1]
empiric_probability.clear()
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics2 = []
empirics2.append(empiric_probability[0])
empirics2.append(empiric_probability[1])
empirics2.append(empiric_probability[2])
empirics2.append(empiric_probability[3])
empirics2.append(empiric_probability[4])


ro = values[2]
empiric_probability.clear()
wait_times, queue_list, queue_time, cancel_list, smo_list = create_SMO(n, lambd, u, ro, m, 10000)
empiric_probability = empiric_probabilities(cancel_list, queue_list, queue_time, wait_times, smo_list, n, m, lambd, u)
theoretic_probability = theoretic_probabilities(n, m, lambd, u, ro)
empirics3 = []
empirics3.append(empiric_probability[0])
empirics3.append(empiric_probability[1])
empirics3.append(empiric_probability[2])
empirics3.append(empiric_probability[3])
empirics3.append(empiric_probability[4])


show_changes(empirics1, empirics2, empirics3, "ro", values)


