import numpy as np


from episode_manager.agent_handler.models.sensor import _to_bev


def main():
    lidar_data = np.load(
        "../expert_data/routes_training_town01_w1_04_19_05_26_20/lidar/0038.npy"
    )

    print("LIDAR DATA SHAPE: ", lidar_data.shape)

    print("LIDAR DATA: ", lidar_data[0])

    return


if __name__ == "__main__":
    main()
