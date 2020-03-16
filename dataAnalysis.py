import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.spatial import distance
import os

from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-deep')

from matplotlib.colors import ListedColormap


def readCSV(filename):
    """
    Function that reads yaw, pitch, roll  for head, left and right controller data from CSV.
    :param filename: the path to the data file.
    :return: a
    """
    df = pd.read_csv(filename)
    return df


def cut_endings(data):
    """
    Delete thr first and last n_samples.
    :param data:
    :return:
    """
    # There are 72 fps. 1 min = 60 s -> 4320 frames.
    # We took a smaller window, of 4000 frames.
    WINDOW_WIDTH = 4000
    data_len = data.shape[0]
    if data_len < WINDOW_WIDTH:
        print("The data has less samples than the WINDOW_WIDTH = ", str(WINDOW_WIDTH))
        return data

    # Fix amount of frames cut in the beginning, as we are always setting the environment.
    cut_beginning = 200
    # Cut more frames at the end because the environment was sometimes closed late.
    cut_end = int(data_len - cut_beginning - WINDOW_WIDTH)

    data = data[cut_beginning:]
    data.reset_index(drop=True, inplace=True)

    data = data[:-cut_end]
    data.reset_index(drop=True, inplace=True)
    return data


def norm(x, min, max):
    return (x - min) / (max - min)


def normalize(data):
    """
    Normalize the data received from the VR headset.
    :param data: A pd.Dataframe that contains the following columns:
        'head_yaw', 'head_pitch', 'head_roll', 'head_x', 'head_y', 'head_z',
        'left_c_yaw', 'left_c_pitch', 'left_c_roll', 'left_c_x', 'left_c_y', 'left_c_z',
        'right_c_yaw', 'right_c_pitch', 'right_c_roll', 'right_c_x', 'right_c_y', 'right_c_z'
    :return:
    """
    
    # 1. Normalize the angles
    # The angles are normalized in order to have the initial position = 0 and varies between [-1, 1]
    # Head angles
    data['head_pitch'] = np.sin(data['head_pitch'] * np.pi / 180.)
    data['head_yaw'] = np.cos(data['head_yaw'] * np.pi / 180.)
    data['head_roll'] = np.sin(data['head_roll'] * np.pi / 180.)

    # Left controller angles
    data['left_c_pitch'] = np.sin(data['left_c_pitch'] * np.pi / 180.)
    data['left_c_yaw'] = np.cos(data['left_c_yaw'] * np.pi / 180.)
    data['left_c_roll'] = np.sin(data['left_c_roll'] * np.pi / 180.)

    # Right controller angles
    data['right_c_pitch'] = np.sin(data['right_c_pitch'] * np.pi / 180.)
    data['right_c_yaw'] = np.cos(data['right_c_yaw'] * np.pi / 180.)
    data['right_c_roll'] = np.sin(data['right_c_roll'] * np.pi / 180.)

    # 2. Normalize X, Y, Z position
    # The values of the X, Y, Z potions are normalized between [0, 1]
    maximum = max(data[['head_x', 'head_y', 'head_z',
                        'left_c_x', 'left_c_y', 'left_c_z',
                        'right_c_x', 'right_c_y', 'right_c_z']].max())
    minimum = min(data[['head_x', 'head_y', 'head_z',
                        'left_c_x', 'left_c_y', 'left_c_z',
                        'right_c_x', 'right_c_y', 'right_c_z']].min())

    # Head position
    data['head_x'] = norm(data['head_x'], minimum, maximum)
    data['head_y'] = norm(data['head_y'], minimum, maximum)
    data['head_z'] = norm(data['head_z'], minimum, maximum)

    # Left controller position
    data['left_c_x'] = norm(data['left_c_x'], minimum, maximum)
    data['left_c_y'] = norm(data['left_c_y'], minimum, maximum)
    data['left_c_z'] = norm(data['left_c_z'], minimum, maximum)

    # Right controller position
    data['right_c_x'] = norm(data['right_c_x'], minimum, maximum)
    data['right_c_y'] = norm(data['right_c_y'], minimum, maximum)
    data['right_c_z'] = norm(data['right_c_z'], minimum, maximum)

    return data


def compute_distance(data_head, data_controller):
    if data_head.shape[0] != data_controller.shape[0]:
        print("ERROR: There are different numbers of samples for head anc controller. Cannot compute distance.")
        return

    dist = []
    for i in range(data_head.shape[0]):
        head_pose = (data_head[data_head.columns[0]][i],
                     data_head[data_head.columns[1]][i],
                     data_head[data_head.columns[2]][i])
        controller_pose = (data_controller[data_controller.columns[0]][i],
                           data_controller[data_controller.columns[1]][i],
                           data_controller[data_controller.columns[2]][i])
        dist.append(distance.euclidean(head_pose, controller_pose))

    return dist


def plot_position(df, labels=['head_x', 'head_y', 'head_z'], title="", filename="", save_fig=False):
    """
    Plot position of a set of data (head or controller) in 3D space.
    :param df: The data that contains the X, Y, Z data.
    :param labels: The labels of the data position 0 - X, 1 - Y, 2 - Z
    :param title: The title of the plot
    :param filename: The filename if the figure should be saved
    :param save_fig: True, if the figure should be saved, False if the figured is shown
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the X, Y, Z values
    xs = df[labels[0]]
    ys = df[labels[1]]
    zs = df[labels[2]]

    ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w', color="red")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    plt.title(title)

    if save_fig:
        plt.savefig(filename)
    else:
        plt.show()


def plot_head_controllers(df, title="", filename= "", save_fig=False):
    """
    Plot the position of the head and the controllers.
    :param df: The pd.Dataframe. It must contan:
        head_x, head_y, head_z
        left_c_x, left_c_y, left_c_z,
        right_c_x, right_c_y right_c_z
    :param filename: The filename where the figure should be saved
    :param save_fig: True, if the figure should be saved, False if the figured is shown
    :return:
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Head and controller position plot")

    ax.scatter(df['head_x'], df['head_y'], df['head_z'], color='red', s=50, alpha=0.6, edgecolors='w', label="Head")
    ax.scatter(df['left_c_x'], df['left_c_y'], df['left_c_z'], color='blue', s=50, alpha=0.6, edgecolors='w', label="Left controller")
    ax.scatter(df['right_c_x'], df['right_c_y'], df['right_c_z'], color='green', s=50, alpha=0.6, edgecolors='w', label="Right controller")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    if save_fig:
        plt.savefig(filename)
    else:
        plt.show()


def plot_all_angles(df, filename="", save_fig=False):
    """
    Create 3 plots, one for each angle (pitch, yaw, roll).
    :param df: A pd.DataFrame. It expects that it has the columns:
        head_pitch, head_yaw, head_roll
    :param filename: The filename where the figure should be saved
    :param save_fig: True, if the figure should be saved, False if the figured is shown
    :return:
    """
    # Take the min number of samples from each axes
    n_samples = min(df['head_pitch'].shape[0], df['head_yaw'].shape[0], df['head_roll'].shape[0])

    pitch_x = np.linspace(0, n_samples, n_samples)
    yaw_x = np.linspace(0, n_samples, n_samples)
    roll_x = np.linspace(0, n_samples, n_samples)

    plt.title(filename)
    plt.subplot(3, 1, 1)

    peaks = {}

    plt.plot(pitch_x, df['head_pitch'], 'red')
    peaks['pitch'], _ = find_peaks(df['head_pitch'], prominence=1, width=20)
    print("Pitch peaks", peaks['pitch'])
    plt.ylabel('Pitch')

    plt.subplot(3, 1, 2)
    plt.plot(yaw_x, df['head_yaw'], 'green')
    peaks['yaw'], _ = find_peaks(df['head_yaw'], prominence=1, width=20)
    print("Yaw peaks", peaks['yaw'])
    plt.ylabel('Yaw')

    plt.subplot(3, 1, 3)
    plt.plot(roll_x, df['head_roll'], 'blue')
    peaks['roll'], _ = find_peaks(df['head_roll'], prominence=1, width=20)
    print("Roll peaks", peaks['roll'])
    plt.ylabel('Roll')

    if save_fig:
        plt.savefig(filename)
    else:
        plt.show()

    return peaks


def plot_frequency_angles(df, title="", filename="", save_fig=False):
    """
    Plot the frequency for the head rotation angles.
    :param df: The DataFrame, it expects to have: "head_pitch", "head_yaw", "head_roll.
    :param title: The title of the figure
    :param filename: The filename where the figure should be saved
    :param save_fig: True, if the figure should be saved, False if the figured is shown
    :return:
    """
    SAMPLING_FREQUENCY = 72
    n_samples = min(df['head_pitch'].shape[0], df['head_yaw'].shape[0], df['head_roll'].shape[0])
    x = np.linspace(0, n_samples, n_samples)

    Pxx = pd.DataFrame(columns=["Pxx_pitch", "Pxx_yaw", "Pxx_roll"])
    freqs = pd.DataFrame(columns=["freqs_pitch", "freqs_yaw", "freqs_roll"])

    ##############   PITCH   #####################
    plt.subplot(321)
    plt.plot(x, df['head_pitch'], color='red')
    plt.xlabel('Time')
    plt.ylabel('Pitch Amplitude')

    # plot the signal in frequency   domain
    plt.subplot(322)
    # sampling frequency = 72 - get a magnitude spectrum
    Pxx["Pxx_pitch"], freqs["freqs_pitch"] = plt.psd(df['head_pitch'], Fs=SAMPLING_FREQUENCY)

    ##############   YAW   #####################
    plt.subplot(323)
    plt.plot(x, df['head_yaw'], color="green")
    plt.xlabel('Time')
    plt.ylabel('Yaw Amplitude')

    # plot the signal in frequency domain
    plt.subplot(324)
    # sampling frequency = 72 - get a magnitude spectrum
    # plt.magnitude_spectrum(df['head_yaw'], Fs=72)
    dt = 0.01
    Pxx["Pxx_yaw"], freqs["freqs_yaw"] = plt.psd(df['head_yaw'], Fs=SAMPLING_FREQUENCY)

    ##############   ROLL   #####################
    plt.subplot(325)
    plt.plot(x, df['head_roll'], color="blue")
    plt.xlabel('Time')
    plt.ylabel('Roll Amplitude')

    # plot the signal in frequency domain
    plt.subplot(326)
    # sampling frequency = 72 - get a magnitude spectrum
    Pxx["Pxx_roll"], freqs["freqs_roll"] = plt.psd(df['head_roll'], Fs=SAMPLING_FREQUENCY)

    # display the plots
    if save_fig:
        plt.savefig(filename)
    else:
        plt.show()

    return Pxx, freqs


def plot_head_controller_distance(data, filename="", save_fig=False):
    """
    Plot the distance between head and the controllers.
    :param data: The DataFrame, it expect to have: "dist_head_right', "dist_head_left"
    :param filename: The filename where the figure should be saved
    :param save_fig: True, if the figure should be saved, False if the figured is shown
    :return:
    """
    n_samples = data.shape[0]
    x = np.linspace(0, n_samples, n_samples)

    # fig = plt.figure(figsize=(8, 6))
    plt.subplot(2,1,1)
    plt.plot(x, data['dist_head_right'], color='red')
    plt.xlabel('Samples')
    plt.ylabel('Right controller - head')

    plt.subplot(2,1,2)
    plt.plot(x, data['dist_head_left'], color='blue')
    plt.xlabel('Samples')
    plt.ylabel('Left controller - head')

    if save_fig:
        plt.savefig(filename)
    else:
        plt.show()


def get_features(df):
    min = df.min()
    max = df.max()
    mean = np.mean(df)
    sd = np.std(df)

    return [min], [max], [mean], [sd]


def compute_angle_features(data):
    """
    Compute basic fetures: min, max, mean, standard deviation for the values of the angles.

    The values for the angles have values between [-1, 1], with 0 being the neutral position.
    :param data: A pd.DataFrame that contains:
        "head_pitch", "head_yaw", "head_roll"
    :return: a pd.DataFrame that contains the contains:
        "min_pitch", "max_pitch", "mean_pitch", "sd_pitch",
        "min_yaw", "max_yaw", "mean_yaw", "sd_yaw",
        "min_roll", "max_roll", "mean_roll", "sd_roll"
    """
    result = pd.DataFrame(columns=["min_pitch", "max_pitch", "mean_pitch", "sd_pitch",
                                   "min_yaw", "max_yaw", "mean_yaw", "sd_yaw",
                                   "min_roll", "max_roll", "mean_roll", "sd_roll"])

    result["min_pitch"], result["max_pitch"], result["mean_pitch"], result["sd_pitch"] = get_features(data["head_pitch"])
    result["min_yaw"], result["max_yaw"], result["mean_yaw"], result["sd_yaw"] = get_features(data["head_yaw"])
    result["min_roll"], result["max_roll"], result["mean_roll"], result["sd_roll"] = get_features(data["head_roll"])

    return result


def get_frequency_features(Pxx, freq):
    """
    Extracts the first four poer spectral density features
    :param Pxx: The values of the power spectral densities
    :param freq: The corresponding frequencies
    :return: A pd.DataFrame that contains:
        "psd0_pitch", "psd1_pitch", "psd2_pitch", "psd3_pitch",
        "psd0_yaw", "psd1_yaw", "psd2_yaw", "psd3_yaw",
        "psd0_roll", "psd1_roll", "psd2_roll", "psd3_roll"
    """
    result = pd.DataFrame(columns=[ "psd0_pitch", "psd1_pitch", "psd2_pitch", "psd3_pitch",
                                    "psd0_yaw", "psd1_yaw", "psd2_yaw", "psd3_yaw",
                                    "psd0_roll", "psd1_roll", "psd2_roll", "psd3_roll"])

    result["psd0_pitch"] = [Pxx["Pxx_pitch"][0]]
    result["psd1_pitch"] = [Pxx["Pxx_pitch"][1]]
    result["psd2_pitch"] = [Pxx["Pxx_pitch"][2]]
    result["psd3_pitch"] = [Pxx["Pxx_pitch"][3]]

    result["psd0_yaw"] = Pxx["Pxx_yaw"][0]
    result["psd1_yaw"] = Pxx["Pxx_yaw"][1]
    result["psd2_yaw"] = Pxx["Pxx_yaw"][2]
    result["psd3_yaw"] = Pxx["Pxx_yaw"][3]

    result["psd0_roll"] = Pxx["Pxx_roll"][0]
    result["psd1_roll"] = Pxx["Pxx_roll"][1]
    result["psd2_roll"] = Pxx["Pxx_roll"][2]
    result["psd3_roll"] = Pxx["Pxx_roll"][3]

    return result


def get_distance_features(data):
    """
    TODO(future improvement): These values can be scaled with the height of the user.
    Computes the basic features min, max, mean, standard deviation
    for the distance between the head and left and right controllers.
    :param data: A pd.DataFrame that contains:
        "dist_head_riht", "dist_head_left"
    :return: A pd.DataFrame that contains:
        "min_dist_right", "max_dist_right", "mean_dist_right", "sd_dist_right",
        "min_dist_left", "max_dist_left", "mean_dist_left", "sd_dist_left"
    """

    result = pd.DataFrame(columns=["min_dist_right", "max_dist_right", "mean_dist_right", "sd_dist_right",
                                   "min_dist_left", "max_dist_left", "mean_dist_left", "sd_dist_left"])

    result["min_dist_right"], result["max_dist_right"], result["mean_dist_right"], result["sd_dist_right"]  = \
        get_features(data["dist_head_right"])

    result["min_dist_left"], result["max_dist_left"], result["mean_dist_left"], result["sd_dist_left"] = \
        get_features(data["dist_head_left"])

    return result


def plot_bar(described):
    mean_cols = ["mean_pitch", "mean_yaw", "mean_roll"]
    sd_cols = ["sd_pitch", "sd_yaw", "sd_roll"]
    mean_vals = pd.DataFrame(columns=mean_cols)
    sd_vals = pd.DataFrame(columns=sd_cols)
    for video_id in range(1,5):
        mean_vals = mean_vals.append(described[video_id-1][1:2][mean_cols])
        sd_vals = sd_vals.append(described[video_id-1][1:2][sd_cols])

    labels = ["Jail break", "Puppies", "Rope", "War"]

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, mean_vals["mean_pitch"], width, label='Pitch')
    rects2 = ax.bar(x , mean_vals["mean_yaw"], width, label='Yaw')
    rects3 = ax.bar(x + width, mean_vals["mean_roll"], width, label='Roll')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean values')
    ax.set_title('Mean rotation angle by video')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


    plt.show()


def main():
    all_results = pd.DataFrame(columns=['min_pitch', 'max_pitch', 'mean_pitch', 'sd_pitch', 'min_yaw',
                                        'max_yaw', 'mean_yaw', 'sd_yaw', 'min_roll', 'max_roll', 'mean_roll',
                                        'sd_roll', 'psd0_pitch', 'psd1_pitch', 'psd2_pitch', 'psd3_pitch',
                                        'psd0_yaw', 'psd1_yaw', 'psd2_yaw', 'psd3_yaw', 'psd0_roll',
                                        'psd1_roll', 'psd2_roll', 'psd3_roll', 'min_dist_right',
                                        'max_dist_right', 'mean_dist_right', 'sd_dist_right', 'min_dist_left',
                                        'max_dist_left', 'mean_dist_left', 'sd_dist_left'])

    do_save = False

    # 1. Read the data.
    current_file = "data_example.csv"
    data = readCSV(current_file)

    # 2.Discard first and least seconds.
    cut_data = cut_endings(data)
    # 3. Normalize data.
    data = normalize(cut_data)

    # 4. Compute the Euclidean distance head - controller
    # Distance between head-left controller
    data["dist_head_right"] = compute_distance(data[['head_x', 'head_y', 'head_z']],
                                               data[['right_c_x', 'right_c_y', 'right_c_z']])

    data["dist_head_left"] = compute_distance(data[['head_x', 'head_y', 'head_z']],
                                              data[['left_c_x', 'left_c_y', 'left_c_z']])

    # Normalize distances:
    maximum = max(data[["dist_head_right", "dist_head_left"]].max())
    minimum = min(data[["dist_head_right", "dist_head_left"]].min())
    data["dist_head_right"] = norm(data["dist_head_right"], minimum, maximum)
    data["dist_head_left"] = norm(data["dist_head_left"], minimum, maximum)

    # 5. Test: Plot positions
    # Plot the head anc controllers position
    plots_path = "plots/"
    if do_save:
        plot_head_controllers(data, filename=plots_path + "_head_controllers.png", save_fig=do_save)

        # Plot the head position only.
        plot_position(data, filename=plots_path + "_head_position.png", save_fig=do_save)

        # Plot all the angles
        plot_all_angles(data, filename=plots_path + "_angles.png", save_fig=do_save)

        # Plot the computed distance between the head and controllers.
        plot_head_controller_distance(data, filename=plots_path + "_dist_h_c.png", save_fig=do_save)

    # 7. Compute simple angle features
    # Compute: min, max, mean, standard deviation
    # PAY ATTENTION TO THE LISTS
    angle_features = compute_angle_features(data)
    # print(angle_features)

    # 8. Plot the frequency for all the angles
    Pxx, freq = plot_frequency_angles(data[['head_pitch', 'head_yaw', 'head_roll']], save_fig=do_save,
                                      filename=plots_path + "freq_angles.png")

    frequency_features = get_frequency_features(Pxx, freq)
    # print(frequency_features)

    # 9. Compute head-controller distance features
    distance_features = get_distance_features(data)
    # print(distance_features)

    # 10. Put together all the features
    # There are in total 32 features:
    # 12 - angle features
    # 12 - angle frequency features
    #  8 - distance head left /right controller features
    # = 32 total features

    result = pd.concat([angle_features, frequency_features, distance_features], axis=1, sort=False)
    # print(result)

    all_results = pd.concat([all_results, result], ignore_index=True)
    # print (all_results)

    # Create new results folder for the user:
    output_path = "output"
    output_filename = "output.csv"
    labels_output_filename = "lables.csv"

    if not(os.path.isdir(output_path)):
        os.mkdir(output_path)

    # 11. Write result to csv
    all_results.to_csv(output_filename, sep=",",  index=False)
    all_results = all_results[0:0]

    # 12. Create label videos file
    labels = pd.DataFrame(columns=["valence"])
    labels["valence"] = [0, 1, 1, 0]
    labels.to_csv(labels_output_filename,  index=False)


if __name__ == '__main__':
    main()
