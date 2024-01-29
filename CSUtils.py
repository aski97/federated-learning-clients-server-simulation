from enum import Enum
import pickle
import struct
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers


class MessageType(Enum):
    FEDERATED_WEIGHTS = 1
    CLIENT_TRAINED_WEIGHTS = 2
    END_FL_TRAINING = 3
    CLIENT_EVALUATION = 4


def get_skeleton_model() -> Sequential:
    """
    Return the skeleton of the model
    :return: skeleton of the model
    """
    return Sequential([
        layers.InputLayer(input_shape=(784,)),
        layers.Dense(10),
        layers.Softmax(),
    ])


def build_message(msg_type: MessageType, body: object) -> bytes:
    """
    Build the message to be sent to:
    First 4 bytes the length of the message,
    then the message in the format of {'type': '', 'body': ''}
    :param MessageType msg_type: type of the message.
    :param object body: body of the message.
    :return: message to be sent to
    :rtype: bytes
    """
    msg = {'type': msg_type, 'body': body}
    msg_serialized = serialize_message(msg)

    # Compute the length of serialized data and convert them in 4 bytes
    length = len(msg_serialized)
    length_bytes = struct.pack("!I", length)

    return length_bytes + msg_serialized


def unpack_message(msg: bytes):
    """
    Unpack the message received from a source.
    The message is formed by two data: type and body of message
    :param bytes msg: buffer of the message received.
    :returns:
        - body : str
            Body of the message.
        - type : MessageType
            Type of the message
    """
    data_deserialized = deserialize_message(msg)

    type = data_deserialized['type']
    body = data_deserialized['body']
    return body, type


def serialize_message(msg):
    return pickle.dumps(msg)


def deserialize_message(msg):
    return pickle.loads(msg)
