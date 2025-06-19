import numpy as np
import open3d as o3d

from utils import deal_files


def find_circumcenter(point_cloud):
    """
    找到点云数据的外心
    :param point_cloud: 点云数据，形状为 (n, 3), n 是点的数量, 3 表示三维坐标
    :return: 外心的坐标，形状为 (3,)
    """
    # 计算点云的均值，即外心（这里简单假设点云均匀分布，外心近似为质心）
    circumcenter = np.mean(point_cloud, axis=0)
    return circumcenter

def transform_point_cloud(xyz):
    """
    将点云转换到以给定外心为新原点的笛卡尔坐标系中
    :param xyz: 点云数据，形状为 (n, 3)
    :return: 转换后的点云数据，形状为 (n, 3)
    """

    # 找到外心
    circumcenter = find_circumcenter(xyz)

    # 从每个点的坐标中减去外心的坐标
    transformed_point_cloud = xyz - circumcenter
    return transformed_point_cloud

def cartesian_to_spherical(xyz):
    """
    将笛卡尔坐标系下的坐标转换为球坐标系下的坐标
    :param xyz: 输入的笛卡尔坐标数组，形状为 (n, 3)
    :return: 球坐标系下的坐标数组，形状为 (n, 3)
    """
    # 提取 x, y, z 坐标
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # 计算球坐标系下的坐标
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    # 组合成球坐标系下的坐标数组
    spherical_coords = np.column_stack((r, theta, phi))
    return spherical_coords

def normalize_r(spherical_coords):
    """
    将球坐标系下的 r 坐标归一化到 [0, 1] 之间
    :param spherical_coords: 球坐标系下的坐标数组，形状为 (n, 3)
    :return: 归一化 r 后的球坐标系坐标数组，形状为 (n, 3)
    """
    r = spherical_coords[:, 0]
    r_min = np.min(r)
    r_max = np.max(r)
    normalized_r = (r - r_min) / (r_max - r_min)
    spherical_coords[:, 0] = normalized_r
    return spherical_coords

def convert_xyz_to_spherical(xyz):
    """
    先将输入的 xyz 坐标原点平移到外接球球心，然后转换为球坐标系下的坐标，最后将 r 归一化到 [0, 1]
    :param xyz: 输入的笛卡尔坐标数组，形状为 (n, 3)
    :return: 球坐标系下的坐标数组，形状为 (n, 3)，且 r 已归一化到 [0, 1]
    """
    # 坐标原点平移到外接球球心
    transformed_point_cloud = transform_point_cloud(xyz)

    # 转换为球坐标系
    spherical_coords = cartesian_to_spherical(transformed_point_cloud)
    # r>=0 theta [0, pi] phi [-pi, pi]

    # 将 r 归一化到 [0, 1]
    normalized_spherical_coords = normalize_r(spherical_coords)

    return normalized_spherical_coords

def process_data(source_file: str, target_file: str) -> None:

    mesh = o3d.io.read_triangle_mesh(source_file)
    point_cloud = mesh.sample_points_uniformly(number_of_points=10000)
    data = np.asarray(point_cloud.points) # (10000, 3) xyz
    spherical_data = convert_xyz_to_spherical(data)
    np.save(file=target_file, arr=spherical_data)

    return


if __name__ == "__main__":

    source_file = "/home/JiXiaoyun/workspace/datasets/ModelNet10"
    target_file = "datasets/ModelNet10_SphereVoxels"
    deal_files(
        source_folder=source_file,
        target_folder=target_file,
        func=process_data
    )

    pass