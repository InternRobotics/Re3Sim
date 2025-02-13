import roboticstoolbox as rtb
import numpy as np

path = "/home/pjlab/main/colmap_handeye/data_collection/shooting_pose_tabletop.txt"


def panda_tcp2hand(joints):
    panda = rtb.models.Panda()
    tcp_pose = panda.fkine(q=joints).A
    hand_pose = panda.fkine(q=joints, end="panda_hand").A
    tcp_2_hand = np.linalg.inv(hand_pose) @ tcp_pose
    return tcp_2_hand


def get_tcp2hand_matrix():
    with open(path, "r") as f:
        lines = f.readlines()
        tcp_2_hands = []
        for line in lines:
            joints = line.split(",")
            joints = [float(joint) for joint in joints]
            joints = np.array(joints).reshape(7)
            tcp_2_hand = panda_tcp2hand(joints)
            # tcp_pose = panda.fkine(q=joints)
            # hand_position = tcp_pose
            # print(hand_position)
            tcp_2_hands.append(tcp_2_hand)
        tcp_2_hands = np.array(tcp_2_hands)
        tcp_2_hand = tcp_2_hands.mean(axis=0)
    return tcp_2_hand


if __name__ == "__main__":
    tcp_2_hand = get_tcp2hand_matrix()
    import json
    from scipy.spatial.transform import Rotation as R

    with open(
        "/isaac-sim/src/assets/calibration/easy_handeye/eye_on_hand/10-17-2.json", "r"
    ) as f:
        data = json.load(f)

    quat = [
        data["rotation"]["x"],
        data["rotation"]["y"],
        data["rotation"]["z"],
        data["rotation"]["w"],
    ]
    trans = [
        data["translation"]["x"],
        data["translation"]["y"],
        data["translation"]["z"],
    ]

    rot_matrix = R.from_quat(quat).as_matrix()

    cam_to_tcp = np.eye(4)
    cam_to_tcp[:3, :3] = rot_matrix
    cam_to_tcp[:3, 3] = trans

    print("TCP to Hand transformation matrix:")
    cam_to_hand = tcp_2_hand @ cam_to_tcp
    translation = cam_to_hand[:3, 3]
    rotation = cam_to_hand[:3, :3]
    print(translation)
    print(R.from_matrix(rotation).as_quat(scalar_first=True))
