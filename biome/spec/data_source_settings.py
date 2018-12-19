# coding: utf-8

"""
    RecognAI API

    Recognai Platform API specification  # noqa: E501

    OpenAPI spec version: v0.1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from biome.spec.data_source_role import DataSourceRole  # noqa: F401,E501
from biome.spec.model_connect import ModelConnect  # noqa: F401,E501


class DataSourceSettings(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'role': 'DataSourceRole',
        'model_connect': 'ModelConnect'
    }

    attribute_map = {
        'role': 'role',
        'model_connect': 'modelConnect'
    }

    def __init__(self, role=None, model_connect=None):  # noqa: E501
        """DataSourceSettings - a model defined in Swagger"""  # noqa: E501

        self._role = None
        self._model_connect = None
        self.discriminator = None

        if role is not None:
            self.role = role
        if model_connect is not None:
            self.model_connect = model_connect

    @property
    def role(self):
        """Gets the role of this DataSourceSettings.  # noqa: E501


        :return: The role of this DataSourceSettings.  # noqa: E501
        :rtype: DataSourceRole
        """
        return self._role

    @role.setter
    def role(self, role):
        """Sets the role of this DataSourceSettings.


        :param role: The role of this DataSourceSettings.  # noqa: E501
        :type: DataSourceRole
        """

        self._role = role

    @property
    def model_connect(self):
        """Gets the model_connect of this DataSourceSettings.  # noqa: E501


        :return: The model_connect of this DataSourceSettings.  # noqa: E501
        :rtype: ModelConnect
        """
        return self._model_connect

    @model_connect.setter
    def model_connect(self, model_connect):
        """Sets the model_connect of this DataSourceSettings.


        :param model_connect: The model_connect of this DataSourceSettings.  # noqa: E501
        :type: ModelConnect
        """

        self._model_connect = model_connect

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(DataSourceSettings, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, DataSourceSettings):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
