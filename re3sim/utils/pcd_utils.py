import numpy as np
import torch

# from pytorch3d.ops import furthest_point_sample
# from pytorch3d.ops import sample_farthest_points as torch3d_ops
import trimesh
import numpy as np

BOUND = None
try:
    import open3d as o3d
except:
    print("Open3D is not installed, some functions will not be available.")

import numpy as np
import roboticstoolbox as rtb

def depth_to_point_cloud(depth, fx, fy, cx, cy):
    """
    Convert depth image to point cloud

    Args:
        depth: depth image (H, W)
        fx, fy: camera focal lengths
        cx, cy: camera principal point coordinates

    Returns:
        points: point cloud (N, 3), where N is the number of valid points
    """
    height, width = depth.shape

    # Create grid coordinates
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)

    # Convert pixel coordinates to normalized plane coordinates
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy

    # Calculate 3D coordinates
    z = depth
    x = x_norm * z
    y = y_norm * z

    # Stack coordinates and reshape
    points = np.stack((x, y, z), axis=-1)
    points = points.reshape(-1, 3)

    # Remove invalid points (depth = 0 or infinity)
    valid = np.logical_and(z > 0, np.isfinite(z))
    points = points[valid.flatten()]

    return points

def smooth_point_cloud(point_cloud, method="MLS", **kwargs):
    """
    Smooth a point cloud.

    Args:
        point_cloud: open3d.geometry.PointCloud to be smoothed
        method: str smoothing method, supports "MLS", "Laplacian", "Bilateral", "Gaussian", "Normal", "Edge-preserving"
        kwargs: optional parameters for different methods

    Returns:
        Smoothed point cloud: open3d.geometry.PointCloud
    """
    if method == "MLS":
        # Moving Least Squares smoothing
        search_param = kwargs.get('search_param', o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        smoothed_pcd = point_cloud.voxel_down_sample(voxel_size=0.02)  # Downsample first
        smoothed_pcd.estimate_normals(search_param=search_param)
        return o3d.geometry.PointCloud.create_from_point_cloud_poisson_disk(smoothed_pcd, kwargs.get('num_points', 30000))

    elif method == "Laplacian":
        # Laplacian smoothing - implemented through triangle mesh
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.1)
        mesh = mesh.filter_smooth_laplacian(number_of_iterations=kwargs.get('iterations', 10))
        return mesh.sample_points_uniformly(number_of_points=len(point_cloud.points))

    elif method == "Bilateral":
        # Estimate normals
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Statistical filtering to remove outliers
        cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=kwargs.get('nb_neighbors', 20),
                                                       std_ratio=kwargs.get('std_ratio', 2.0))
        inlier_cloud = point_cloud.select_by_index(ind)
        
        # Voxel downsampling to smooth point cloud
        voxel_size = kwargs.get('voxel_size', 0.05)
        smoothed_pcd = inlier_cloud.voxel_down_sample(voxel_size)
        
        # Re-estimate normals
        smoothed_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        return smoothed_pcd

    elif method == "Gaussian":
        # Gaussian smoothing - implemented through triangle mesh
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(point_cloud, alpha=0.1)
        mesh = mesh.filter_smooth_simple(number_of_iterations=kwargs.get('iterations', 5))
        return mesh.sample_points_uniformly(number_of_points=len(point_cloud.points))

    elif method == "Normal":
        # Normal vector smoothing
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        normals = np.asarray(point_cloud.normals)
        smoothed_normals = o3d.geometry.PointCloud.compute_nearest_neighbor_distance(point_cloud)
        point_cloud.normals = o3d.utility.Vector3dVector(smoothed_normals)
        return point_cloud

    elif method == "Edge-preserving":
        raise NotImplementedError("Edge-preserving smoothing is not directly supported in Open3D.")

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

def point_cloud_to_mesh(points: np.ndarray, method="ball_pivoting", radii=None, smooth_iterations=5, depth=8):
    """
    Convert point cloud to triangle mesh, supporting Ball-Pivoting algorithm and Poisson reconstruction.

    Args:
        points (np.ndarray): Point cloud numpy array with shape (N, 3)
        method (str): Specified algorithm, "ball_pivoting" or "poisson"
        radii (list): Radius list for Ball-Pivoting algorithm, default [0.005, 0.01, 0.02, 0.04]
        smooth_iterations (int): Number of mesh smoothing iterations when using Ball-Pivoting. Default 5
        depth (int): Depth parameter for Poisson reconstruction, controls mesh resolution. Default 8

    Returns:
        o3d.geometry.TriangleMesh: Generated triangle mesh
    """
    if method not in ["ball_pivoting", "poisson"]:
        raise ValueError("method must be 'ball_pivoting' or 'poisson'")

    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate normals (required for both algorithms)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    if method == "ball_pivoting":
        if radii is None:
            radii = [0.005, 0.01, 0.02, 0.04]

        # Generate mesh using Ball-Pivoting algorithm
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))

        # Mesh smoothing
        mesh = mesh.filter_smooth_simple(number_of_iterations=smooth_iterations)

    elif method == "poisson":
        # Generate mesh using Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    return mesh


def bbox_from_pcd(pcd):
    """
    Get the bounding box of a point cloud.

    Args:
        pcd (np.ndarray): The point cloud data with shape (N, 3).

    Returns:
        tuple: A tuple containing the minimum and maximum coordinates of the bounding box.
    """
    return np.min(pcd, axis=0), np.max(pcd, axis=0)


def pcd_filter_bound(cloud, eps=1e-3, max_dis=1.5, bound=BOUND):
    # return (
    #     (pcd["pos"][..., 2] > eps)
    #     & (pcd["pos"][..., 1] < max_dis)
    #     & (pcd["pos"][..., 0] < max_dis)
    #     & (pcd["pos"][..., 2] < max_dis)
    # )
    if isinstance(cloud, dict):
        pc = cloud["pos"]  # (n, 3)
    else:
        assert isinstance(cloud, np.ndarray), f"{type(cloud)}"
        assert cloud.shape[1] == 3, f"{cloud.shape}"
        pc = cloud

    # remove robot table
    within_bound_x = (pc[..., 0] > bound[0]) & (pc[..., 0] < bound[1])
    within_bound_y = (pc[..., 1] > bound[2]) & (pc[..., 1] < bound[3])
    within_bound_z = (pc[..., 2] > bound[4]) & (pc[..., 2] < bound[5])
    within_bound = np.nonzero(
        np.logical_and.reduce((within_bound_x, within_bound_y, within_bound_z))
    )[0]

    return within_bound


def pcd_filter_with_mask(obs, mask, env=None):
    assert isinstance(obs, dict), f"{type(obs)}"
    for key in ["pos", "colors", "seg", "visual_seg", "robot_seg"]:
        select_mask(obs, key, mask)


def pcd_downsample(
    obs,
    env=None,
    bound_clip=False,
    ground_eps=-1e-3,
    max_dis=15,
    num=1200,
    method="fps",
    bound=BOUND,
):
    assert method in [
        "fps",
        "uniform",
    ], "expected method to be 'fps' or 'uniform', got {method}"

    sample_mehod = uniform_sampling if method == "uniform" else fps_sampling

    if bound_clip:
        pcd_filter_with_mask(
            obs,
            pcd_filter_bound(obs, eps=ground_eps, max_dis=max_dis, bound=bound),
            env,
        )
    pcd_filter_with_mask(obs, sample_mehod(obs["pos"], num), env)
    return obs


def fps_sampling(points, npoints=1200):
    num_curr_pts = points.shape[0]
    if num_curr_pts < npoints:
        return np.random.choice(num_curr_pts, npoints, replace=True)
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    try:
        fps_idx = furthest_point_sample(points[..., :3], npoints)
    except:
        npoints = torch.tensor([npoints]).cuda()
        _, fps_idx = torch3d_ops.sample_farthest_points(points[..., :3], K=npoints)

    return fps_idx.squeeze(0).cpu().numpy()


def uniform_sampling(points, npoints=1200):
    n = points.shape[0]
    index = np.arange(n)
    if n == 0:
        return np.zeros(npoints, dtype=np.int64)
    if index.shape[0] > npoints:
        np.random.shuffle(index)
        index = index[:npoints]
    elif index.shape[0] < npoints:
        num_repeat = npoints // index.shape[0]
        index = np.concatenate([index for i in range(num_repeat)])
        index = np.concatenate([index, index[: npoints - index.shape[0]]])
    return index


def add_gaussian_noise(
    cloud: np.ndarray, np_random: np.random.RandomState, noise_level=1
):
    # cloud is (n, 3)
    num_points = cloud.shape[0]
    multiplicative_noise = (
        1 + np_random.randn(num_points)[:, None] * 0.01 * noise_level
    )  # (n, 1)
    return cloud * multiplicative_noise


def add_perlin_noise(
    points, scale=0.1, octaves=1, persistence=0.5, lacunarity=2.0, amplitude=1.0
):
    """
    Adds Perlin noise to a point cloud.

    :param points: A numpy array of shape (n, 3) representing the point cloud.
    :param scale: Scale of the Perlin noise.
    :param octaves: Number of octaves for the Perlin noise.
    :param persistence: Persistence of the Perlin noise.
    :param lacunarity: Lacunarity of the Perlin noise.
    :param amplitude: Amplitude of the noise to make the effect more noticeable.
    :return: A numpy array of the same shape as points with added Perlin noise.
    """
    noisy_points = np.zeros_like(points)

    for i, point in enumerate(points):
        x, y, z = point
        noise_x = (
            noise.pnoise3(
                x * scale,
                y * scale,
                z * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noise_y = (
            noise.pnoise3(
                y * scale,
                z * scale,
                x * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noise_z = (
            noise.pnoise3(
                z * scale,
                x * scale,
                y * scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
            )
            * amplitude
        )
        noisy_points[i] = point + np.array([noise_x, noise_y, noise_z])

    return noisy_points


def dbscan_outlier_removal(pcd):  # (N, 3)
    clustering = DBSCAN(eps=0.1, min_samples=10).fit(pcd)
    labels = clustering.labels_
    print("Number of clusters: ", len(set(labels)))
    # max_label = max(set(labels), key=labels) # only keep the cluster with the most points

    return np.array(pcd)[labels != -1]


def open3d_pcd_outlier_removal(
    pointcloud, radius_nb_num=300, radius=0.08, std_nb_num=300, vis=False
):
    """N x 3 or N x 6"""
    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(pointcloud[..., :3])
    model_pcd.colors = o3d.utility.Vector3dVector(pointcloud[..., 3:])
    # prior: it's a single rigid object
    model_pcd.remove_duplicated_points()
    model_pcd.remove_non_finite_points()

    cl, ind = model_pcd.remove_radius_outlier(
        nb_points=int(radius_nb_num), radius=radius
    )
    model_pcd.points = o3d.utility.Vector3dVector(np.array(model_pcd.points)[ind, :3])
    model_pcd.colors = o3d.utility.Vector3dVector(np.array(model_pcd.colors)[ind, :3])

    cl, ind = model_pcd.remove_statistical_outlier(
        nb_neighbors=std_nb_num, std_ratio=2.0
    )

    if vis:
        display_inlier_outlier(model_pcd, ind)
    # return pointcloud[ind] # No remove, not sure why
    return np.array(model_pcd.select_by_index(ind).points), np.array(
        model_pcd.select_by_index(ind).colors
    )


# TODO: not correct
def render_point_cloud_to_image(point_cloud_np, intrinsic, extrinsic, width, height):
    """
    Render the point cloud to an RGB and depth image using Open3D.

    Parameters:
    - point_cloud_np (np.ndarray): The input point cloud to render (n x 3 numpy array).
    - intrinsic (dict): Camera intrinsic parameters. It should contain:
        - fx: Focal length in x direction
        - fy: Focal length in y direction
        - cx: Principal point x
        - cy: Principal point y
    - extrinsic (dict): Camera extrinsic parameters. It should contain:
        - position: Camera position as (x, y, z)
        - rotation: Rotation matrix (3x3 numpy array)
    - width (int): Width of the output image
    - height (int): Height of the output image

    Returns:
    - rgb_image (numpy.ndarray): Rendered RGB image
    - depth_image (numpy.ndarray): Rendered depth image
    """
    # Convert the numpy point cloud to an Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_np)

    # Create an Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    # Add the point cloud to the visualizer
    vis.add_geometry(point_cloud)

    # Set up the camera intrinsic parameters
    intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        intrinsic["fx"],
        intrinsic["fy"],
        intrinsic["cx"],
        intrinsic["cy"],
    )

    # Set up the camera extrinsic parameters (view matrix)
    cam_pos = extrinsic["position"]
    cam_rot = extrinsic["rotation"]

    adjusted_cam_rot = cam_rot
    # Open3D's view control uses a camera extrinsic matrix (4x4), which combines rotation and translation
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = adjusted_cam_rot
    extrinsic_matrix[:3, 3] = cam_pos

    # Create PinholeCameraParameters and set its intrinsic and extrinsic
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic_matrix
    camera_params.extrinsic = extrinsic_matrix

    # Set up the view control
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    # Capture the depth image
    vis.poll_events()
    vis.update_renderer()
    depth_image = vis.capture_depth_float_buffer(True)

    # Capture the RGB image
    vis.poll_events()
    vis.update_renderer()
    rgb_image = vis.capture_screen_float_buffer(True)

    # Convert images to numpy arrays
    depth_image = np.asarray(depth_image)
    rgb_image = np.asarray(rgb_image)

    # Destroy the visualizer
    vis.destroy_window()

    return rgb_image, depth_image
