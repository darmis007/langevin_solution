import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse


def CreateParser():
    '''Creating a parser for command line options for this simulator'''
    parser = argparse.ArgumentParser(
        description="Inputs to get Langevin Simulator started")
    parser.add_argument('-x0', '--initial_position', nargs='?', type=float,
                        default=1, help='Initial position of the molecule, default = 1')
    parser.add_argument('-v0', '--initial_velocity', nargs='?', type=float,
                        default=0.1, help='Velocity of particle at initial position, default = 0.1')
    parser.add_argument('-temp', '--temperature', nargs='?', type=float,
                        default=1, help='Temperature at which simulation runs, default = 1')
    parser.add_argument('-dc', '--damping_coeff', nargs='?', type=float,
                        default=1, help='Damping coefficient for the system, default = 1')
    parser.add_argument('-dt', '--time_step', nargs='?', type=float,
                        default=0.1, help='Time step for simulation, default = 0.1')
    parser.add_argument('-t', '--total_time', nargs='?', type=float, default=10,
                        help='Total time for which simulation should run, default = 1')
    parser.add_argument('--input_file', nargs='?', type=str, default='./Lans/Pot_example.txt',
                        help='Specify the path of input file, default = \'./Lans/Pot_example.txt\'')
    parser.add_argument('--out_file', nargs='?', type=str, default='./Lans/output.txt',
                        help='Specify file path where you want output, default = \'./Lans/output.txt\' ')
    return parser

def read_potential_energy_file(input_file):
    try:
        data = np.transpose(np.genfromtxt(
            input_file, delimiter=' ', skip_header=1, dtype=float))
        return data[1], data[2], data[3]
    except:
        print('Please enter correct filepath')


def random_force(T, gamma, degree='K',kb=1):
    '''
    By putting the arguments Temperature,Damping Coefficient,Time Step we will get random Force
    '''
    if degree == 'C':
        T = T+273.15
    if degree == 'F':
        T = ((T - 32)/1.8) + 273.15
    random_force_generated = np.random.normal(
        loc=0, scale=(2*T*gamma*kb)**0.5, size=1)
    #We calculate the variance of the random force
    return float(random_force_generated)


def drag_force(gamma, v):
    '''
    calculate the resistance force offered by the medium
    '''
    drag = -1*gamma*v

    return drag


def potential_force(x, pos, energy):
    '''
    calculate potential force at a point x 
    '''
    potential_force = np.interp(x, pos, energy)
    return potential_force


def euler(position, velocity, gamma, temperature, position_array, energy_array, time_step=1, degree='K', mass=1):
    '''
    calculate position,velocity and acceleration at a certain time_step
    '''
    force = drag_force(gamma, velocity) + random_force(temperature, gamma,degree='K') - potential_force(position, position_array, energy_array)
    acceleration = (force/mass)
    velocity += acceleration*time_step
    position += velocity*time_step
    return position, velocity, acceleration


def write_output(out_file, output):
    f = open(out_file, 'w')
    f.write('index time position velocity\n')
    for line in output:
        for i in range(len(line)):
            if(i == 0):
                f.write('{} '.format(line[i]))
            else:
                f.write('{:.4f} '.format(line[i]))
        f.write('\n')
    f.close()

def main(sv): 
    '''Run simulation and send real-time position to visualization'''
    kb = 1
    parser = CreateParser()
    args = parser.parse_args()
    if(args.damping_coeff <= 0 or args.temperature <= 0 or args.time_step <= 0 or args.total_time <= 0):
        #damping coefficient, temperature and time must be positive
        raise ValueError(
            "Damping coefficient, temperature, time_step and total_time must be non-zero positive values")

    # assigning values from namespace to variables
    x0, v0, temperature, gamma, time_step, total_time, input_file, out_file = args.initial_position, args.initial_velocity, args.temperature, args.damping_coeff, args.time_step, args.total_time, args.input_file, args.out_file
    N = int(total_time/time_step)
    position_array, force_array, energy_array = read_potential_energy_file(input_file)
    position = x0
    velocity = v0
    time_elapsed = 0
    count = 0
    output = []
    plt.rcParams['animation.html'] = 'jshtml'
    velocity_figure = plt.figure()
    acceleration_figure = plt.figure()
    position_time_plot = velocity_figure.add_subplot(111)
    velocity_time_plot = acceleration_figure.add_subplot(222)
    velocity_figure.show()
    x, y = [], []
    n = []
    for i in range(N):
        new_pos, new_vel, new_acc = euler(position, velocity, gamma, temperature, position_array, energy_array, time_step)
        time_elapsed += time_step
        count += 1
        output.append([count, time_elapsed, new_pos, new_vel])
        position = new_pos
        velocity = new_vel
        x.append(time_elapsed)
        y.append(position)
        n.append(velocity)
        position_time_plot.plot(x, y, color='b')
        position_time_plot.set_xlabel('Time')
        position_time_plot.set_ylabel('Position')
        velocity_time_plot.plot(x, n, color='r')
        velocity_time_plot.set_xlabel('Time')
        velocity_time_plot.set_ylabel('Velocity')
        velocity_figure.canvas.draw()
        acceleration_figure.canvas.draw()
        time.sleep(0.001)
write_output('tmkoc',out_file)
print('Final position = {:.4f}, Final velocity = {:.4f}'.format(position, velocity))
    


