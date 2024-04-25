# -*- coding: UTF8 -*-
"""
Coordinate frame handling
author: Joshua Bird

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging
import typing
from typing import DefaultDict, List

import numpy as np
import networkx as nx
from rosbags.rosbag1 import Reader as Rosbag1Reader
from rosbags.rosbag2 import Reader as Rosbag2Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr, serialize_cdr
# from rosbags.serde.serdes import deserialize_cdr, ros1_to_cdr
from scipy.spatial.transform import Rotation


from evo import EvoException
from evo.tools.settings import SETTINGS

logger = logging.getLogger(__name__)

SUPPORTED_TRANSFORM_MSGS = [
    "interfaces/msg/Sim3TransformStamped", "tf2_msgs/msg/TFMessage"]


class CoordinateFrameCacheException(EvoException):
    pass


class CoordinateFrameDecoder(object):
    def __init__(self, reader: typing.Union[Rosbag1Reader, Rosbag2Reader], topic: str = "/sim3_transform") -> None:
        """

        """
        self.coord_frames = {}
        self.coord_frames_graph = nx.DiGraph()

        if topic not in reader.topics:
            raise CoordinateFrameCacheException(
                "no messages for topic {} in bag".format(topic))

        connections = [
            c for c in reader.connections if c.topic == topic
        ]

        for connection, _, rawdata in reader.messages(
                connections=connections):
            if connection.msgtype not in SUPPORTED_TRANSFORM_MSGS:
                raise CoordinateFrameCacheException(
                    f"Expected {SUPPORTED_TRANSFORM_MSGS} message type for topic "
                    f"{topic}, got: {connection.msgtype}")

            if isinstance(reader, Rosbag1Reader):
                msg = deserialize_cdr(ros1_to_cdr(rawdata, connection.msgtype),
                                      connection.msgtype)
            else:
                msg = deserialize_cdr(rawdata, connection.msgtype)

            if connection.msgtype == "interfaces/msg/Sim3TransformStamped":
                transformation_matrix = sim3_msg_to_matrix(msg.transform)

                if msg.child_frame_id not in self.coord_frames:
                    self.coord_frames[msg.child_frame_id] = []
                    self.coord_frames_graph.add_edge(
                        msg.header.frame_id, msg.child_frame_id
                    )

                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.coord_frames[msg.child_frame_id].append(
                    (timestamp, transformation_matrix)
                )
            if connection.msgtype == "tf2_msgs/msg/TFMessage":
                for transformStamped in msg.transforms:
                    transformation_matrix = se3_msg_to_matrix(
                        transformStamped.transform)

                    if transformStamped.child_frame_id not in self.coord_frames:
                        self.coord_frames[transformStamped.child_frame_id] = []
                        self.coord_frames_graph.add_edge(
                            transformStamped.header.frame_id, transformStamped.child_frame_id
                        )

                    timestamp = transformStamped.header.stamp.sec + \
                        transformStamped.header.stamp.nanosec * 1e-9
                    self.coord_frames[transformStamped.child_frame_id].append(
                        (timestamp, transformation_matrix)
                    )

        if next(nx.simple_cycles(self.coord_frames_graph), None) is not None:
            raise CoordinateFrameCacheException(
                "Coordinate frame graph is not a directed acyclic graph"
            )

        for frame_id in self.coord_frames:
            self.coord_frames[frame_id].sort(key=lambda x: x[0])

    def get_transformation(self, base_frame_id: str, target_frame_id: str, timestamp: float = None) -> np.ndarray:
        """
        Get the transformation matrix from the base frame to the target frame
        """
        try:
            path = nx.shortest_path(
                self.coord_frames_graph, source=base_frame_id, target=target_frame_id)
        except:
            raise CoordinateFrameCacheException(
                f"No path from {base_frame_id} to {target_frame_id}"
            )

        transformation_matrix = np.eye(4)

        for frame_id in path[1:]:
            frame_matrix = self.get_frame_at_timestamp(frame_id, timestamp)
            transformation_matrix = frame_matrix @ transformation_matrix

        return transformation_matrix

    def get_frame_at_timestamp(self, frame_id: str, timestamp: float = None) -> np.ndarray:
        """
        Get the transformation matrix for a frame at a given timestamp
        """
        if timestamp is None:
            return self.coord_frames[frame_id][-1][1]

        for i in range(len(self.coord_frames[frame_id])):
            if self.coord_frames[frame_id][i][0] > timestamp:
                return self.coord_frames[frame_id][i - 1][1]

        raise CoordinateFrameCacheException(
            f"No transformation matrix for frame {frame_id} at timestamp {timestamp}"
        )


def sim3_msg_to_matrix(sim3_transform) -> np.ndarray:
    """
    Convert a Sim3Transform to a 4x4 matrix
    """
    quaternion = Rotation.from_quat([
        sim3_transform.rotation.x,
        sim3_transform.rotation.y,
        sim3_transform.rotation.z,
        sim3_transform.rotation.w
    ])
    rotation_matrix = quaternion.as_matrix()

    # Construct the scaling matrix
    scale_matrix = np.eye(3) * sim3_transform.scale

    # Combine rotation and scaling
    rotation_scale_matrix = np.dot(rotation_matrix, scale_matrix)

    # Create the 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_scale_matrix
    transform_matrix[:3, 3] = [
        sim3_transform.translation.x,
        sim3_transform.translation.y,
        sim3_transform.translation.z
    ]

    return transform_matrix


def se3_msg_to_matrix(sim3_transform) -> np.ndarray:
    """
    Convert a SE3Transform to a 4x4 matrix
    """
    quaternion = Rotation.from_quat([
        sim3_transform.rotation.x,
        sim3_transform.rotation.y,
        sim3_transform.rotation.z,
        sim3_transform.rotation.w
    ])
    rotation_matrix = quaternion.as_matrix()

    # Create the 4x4 transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [
        sim3_transform.translation.x,
        sim3_transform.translation.y,
        sim3_transform.translation.z
    ]

    return transform_matrix
