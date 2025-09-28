import ast
import copy
import logging
import argparse
import os
import random
import pandas as pd
import torch
import sys
sys.path.append("..")
sys.path.append("../fourier/src")
import architectures.constrained1d as arch
import utilities.scheduling as scheduling
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import utilities.analytical_both_ends_fixed
from differentiation.differentiate import first_derivative, second_derivative

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    randomSeed,
    initialProfile,
    architecture,
    startingNeuralNetwork,
    duration,
    alpha,
    scheduleFilepath,
    numberOfDiffEquResPoints,
    displayResults
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    logging.info(f"train.main(); device = {device}; architecture = {architecture}")

    outputDirectory += "_" + architecture
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    # Load the initial temperature profile
    initial_profile_df = pd.read_csv(initialProfile)
    x_u0 = initial_profile_df.values  # (N_initial, 2)
    xs = x_u0[:, 0]
    length = xs[-1]
    xs_tsr = torch.tensor(xs).unsqueeze(1)/length  # (N_initial, 1), normalized values [0, 1]
    xs_0_tsr = torch.cat([xs_tsr, torch.zeros_like(xs_tsr)], dim=1).float().to(device)  # (N_initial, 2)

    u0 = x_u0[:, 1]
    u0_tsr = torch.tensor(u0).float()  # (N_initial)
    u0_tsr = u0_tsr.unsqueeze(1).to(device)  # (N_initial, 1)

    # Create the neural network
    neural_net = None
    architecture_tokens = architecture.split('_')
    if architecture_tokens[0] == 'HardConstrained1dPinn':
        neural_net = arch.HardConstrained1dPinn(
            number_of_blocks=int(architecture_tokens[1]),
            block_width=int(architecture_tokens[2]),
            number_of_outputs=int(architecture_tokens[3]),
            initial_profile=u0,
            time_bubble_beta=float(architecture_tokens[4])
        )
    elif architecture_tokens[0] == 'HardConstrained1dMLP':
        neural_net = arch.HardConstrained1dMLP(
            layer_widths=ast.literal_eval(architecture_tokens[1]),
            number_of_outputs=int(architecture_tokens[2]),
            initial_profile=u0,
            time_bubble_beta=float(architecture_tokens[3])
        )
    elif architecture_tokens[0] == 'HardConstrained1dResNet':
        neural_net = arch.HardConstrained1dResNet(
            number_of_blocks=int(architecture_tokens[1]),
            block_width=int(architecture_tokens[2]),
            number_of_outputs=int(architecture_tokens[3]),
            initial_profile=u0,
            time_bubble_beta=float(architecture_tokens[4])
        )
    else:
        raise NotImplementedError(f"train.main(): Not implemented architecture '{architecture}'")
    if startingNeuralNetwork is not None:
        neural_net.load_state_dict(torch.load(startingNeuralNetwork))
    neural_net.to(device)

    # Test the initial profile interpolation
    increasing_x_tsr = torch.linspace(0, 1, 1001).unsqueeze(1).to(device)  # (B, 1)
    initial_profile_interpolation_tsr = neural_net.initial_profile_interpolator.batch_evaluate(increasing_x_tsr)
    fig, ax = plt.subplots()
    ax.plot(length * increasing_x_tsr.cpu().numpy(), initial_profile_interpolation_tsr.cpu().numpy(), label='interpolation')
    ax.scatter(x_u0[:, 0], x_u0[:, 1], label='samples', c='red')
    # Test the initial prediction
    predicted_u_tsr = neural_net(torch.cat([increasing_x_tsr, torch.zeros_like(increasing_x_tsr)], dim=1))
    ax.plot(length * increasing_x_tsr.cpu().detach().numpy(), predicted_u_tsr.cpu().detach().numpy(), label='predicted u0', c='green')
    ax.legend()
    plt.savefig(os.path.join(outputDirectory, "initial_profile_interpolation.png"))
    plt.close()


    # Load the schedule
    schedule_df = pd.read_csv(scheduleFilepath)
    schedule = scheduling.Schedule(schedule_df)

    # Training parameters
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=schedule.parameters(1)['learning_rate'], weight_decay=0)

    # Training loop
    minimum_loss = float('inf')
    champion_neural_net = None
    phase = schedule.parameters(1)['phase']
    diff_eqn_residual_x_t_tsr = 1.0 * torch.rand((numberOfDiffEquResPoints, 2), requires_grad=True).to(
        device)  # (N_res, 2), normalized dimensions

    with open(os.path.join(outputDirectory, "epochLoss.csv"), 'w') as epoch_loss_file:
        epoch_loss_file.write("epoch,loss,is_champion\n")
        for epoch in range(1, schedule.last_epoch() + 1):
            # Set the neural network to training mode
            neural_net.train()
            neural_net.zero_grad()

            current_parameters = schedule.parameters(epoch)
            if current_parameters['phase'] != phase:
                phase = current_parameters['phase']
                logging.info(f" ---- Phase {phase} ----")
                optimizer = torch.optim.Adam(neural_net.parameters(), lr=current_parameters['learning_rate'],
                                             weight_decay=0)
                #neural_net = champion_neural_net
                diff_eqn_residual_x_t_tsr = 1.0 * torch.rand((numberOfDiffEquResPoints, 2), requires_grad=True).to(
                    device)  # (N_res, 2), normalized dimensions

            # Differential equation residual loss
            diff_eqn_residual_x_t_tsr.detach_()
            diff_eqn_residual_x_t_tsr.requires_grad = True
            du_dx__du_dt = first_derivative(neural_net, diff_eqn_residual_x_t_tsr)
            du_dt = du_dx__du_dt[:, 1]  # (N_res)

            d2u_dx2__d2u_dxdt = second_derivative(neural_net, diff_eqn_residual_x_t_tsr, 0)  # (N_res, 2)
            d2u_dx2 = d2u_dx2__d2u_dxdt[:, 0]  # (N_res)
            diff_eqn_residual = 1.0/duration * du_dt - alpha/length**2 * d2u_dx2  # (N_res)
            diff_eqn_residual_loss = criterion(diff_eqn_residual, torch.zeros_like(diff_eqn_residual))

            loss = diff_eqn_residual_loss
            is_champion = False
            if loss.item() < minimum_loss:
                minimum_loss = loss.item()
                champion_neural_net = copy.deepcopy(neural_net)
                is_champion = True
                champion_filepath = os.path.join(outputDirectory, f"{architecture}.pth")
                torch.save(champion_neural_net.state_dict(), champion_filepath)

            logging.info(f"Epoch {epoch}: loss = {loss.item()}")
            if is_champion:
                logging.info(f" **** Champion! ****")
            epoch_loss_file.write(f"{epoch},{loss.item()},{is_champion}\n")

            loss.backward()
            optimizer.step()

    logging.info(f"minimum_loss = {minimum_loss}")

    if displayResults:
        u_analytical = utilities.analytical_both_ends_fixed.main(alpha, duration,
                                                       256, initialProfile)

        u = np.zeros((256, xs_tsr.shape[0]), dtype=float)
        delta_T = 1.0/(256 - 1)
        images = []
        for t_ndx in range(256):
            t = t_ndx * delta_T  # [0 ... 1.0]
            x_t = torch.zeros(xs_tsr.shape[0], 2).to(device)
            x_t[:, 0] = xs_tsr[:, 0]  # Normalized x
            x_t[:, 1] = t
            u_t = champion_neural_net(x_t)  # (N_spatial, 1)
            u[t_ndx, :] = u_t[:, 0].cpu().detach().numpy()
            fig, ax = plt.subplots()
            ax.set_ylabel("Temperature (°C)")
            ax.set_xlabel("x (m)")
            ax.plot(xs_tsr[:, 0].cpu().detach().numpy() * length, u[t_ndx, :], label="Prediction")
            ax.plot(xs_tsr[:, 0].cpu().detach().numpy() * length, u_analytical[t_ndx, :], label="Analytical")
            ax.legend(loc="upper right")
            ax.set_title(f"t = {(t * duration):.1f} s")
            plt.ylim(20, 160)
            plt.grid()
            image_filepath = os.path.join(outputDirectory, f"prediction_{t_ndx}.png")
            plt.savefig(image_filepath)
            plt.close()  # Prevents the figure from being displayed later
            images.append(imageio.imread(image_filepath))
            try:
                os.remove(image_filepath)
            except Exception as e:
                raise RuntimeError(f"train.main(): Error caught while trying to erase {image_filepath}: {e}")


        imageio.mimsave(os.path.join(outputDirectory, "predictions.gif"), images, duration=30, loop=0)

        ts = np.arange(0, 1.0 + delta_T/2, delta_T)
        xs = xs_tsr[:, 0].numpy()
        fig, ax = plt.subplots(3, 1)
        c = ax[0].pcolormesh(length * xs, duration * ts, u, cmap='viridis')
        ax[0].axis([0, length, 0, duration])
        ax[0].set_xlabel("x (m)")
        ax[0].set_ylabel("t (s)")
        ax[0].set_title("Prediction")
        fig.colorbar(c, ax=ax[0])

        c = ax[1].pcolormesh(length * xs, duration * ts, u_analytical, cmap='viridis')
        ax[1].axis([0, length, 0, duration])
        ax[1].set_xlabel("x (m)")
        ax[1].set_ylabel("t (s)")
        ax[1].set_title("Analytical solution")
        fig.colorbar(c, ax=ax[1])

        u0_predicted = u[0, :]
        u0_analytical = u_analytical[0, :]
        ax[2].plot(xs * length, u0, label='u0')
        ax[2].plot(xs * length, u0_predicted, label='u0 predicted')
        ax[2].plot(xs * length, u0_analytical, label='u0 analytical')
        ax[2].legend()
        ax[2].grid()

        plt.show()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_train'", default='./output_train')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--initialProfile', help="The initial temperature profile. Default: './heated_segments.csv'", default='./heated_segments.csv')
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'HardConstrained1dPinn_2_32_1_2'", default='HardConstrained1dPinn_2_32_1_2')
    parser.add_argument('--startingNeuralNetwork', help="The filepath to the starting neural network. Default: 'None'", default='None')
    parser.add_argument('--duration', help="The simulation duration, in seconds. Default: 10.0", type=float, default=10.0)
    parser.add_argument('--alpha', help="The material's diffusivity, in m^2/s. Default: 0.0001", type=float, default=0.0001)
    parser.add_argument('--scheduleFilepath', help="The filepath to the training schedule. Default: './schedule.csv'", default='./schedule.csv')
    parser.add_argument('--numberOfDiffEquResPoints', help="The number of points for the differential equation residual. Default: 256", type=int, default=256)
    parser.add_argument('--displayResults', help="Display the results", action='store_true')
    args = parser.parse_args()
    if args.startingNeuralNetwork.upper() == 'NONE':
        args.startingNeuralNetwork = None

    main(
        args.outputDirectory,
        args.randomSeed,
        args.initialProfile,
        args.architecture,
        args.startingNeuralNetwork,
        args.duration,
        args.alpha,
        args.scheduleFilepath,
        args.numberOfDiffEquResPoints,
        args.displayResults
    )